using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Xml.Linq;

namespace RobotSlamServer
{
    public class StereoFrameRequest
    {
        public double RobotX { get; set; }
        public double RobotY { get; set; }
        // Углы поворота из Unity
        public double RobotAngleX { get; set; }
        public double RobotAngleY { get; set; }
        public double RobotAngleZ { get; set; }
        public double CartographerSpeed { get; set; }
        public CameraInfo LeftCameraInfo { get; set; }
        public byte[] LeftImageData { get; set; }
        public CameraInfo RightCameraInfo { get; set; }
        public byte[] RightImageData { get; set; }
    }

    public class StereoFrameResponse
    {
        public byte[] ProcessedLeftImage { get; set; }
        public byte[] ProcessedRightImage { get; set; }
        public byte[] SlamMapImage { get; set; }
        public float NextX { get; set; }
        public float NextY { get; set; }
        public float NextAngleX { get; set; }
        public float NextAngleY { get; set; }
        public float NextAngleZ { get; set; }
        public bool IsSlammingComplete { get; set; }
    }

    // Обновленный класс CameraInfo с дополнительными данными позиции камеры.
    public class CameraInfo
    {
        // Фокусное расстояние камеры
        public double FocalLength { get; set; }

        // Координаты главной точки на матрице камеры
        public double PrincipalPointX { get; set; }
        public double PrincipalPointY { get; set; }

        // Смещение камеры относительно центра родительского элемента по оси X и Z.
        public double OffsetX { get; set; }
        public double OffsetZ { get; set; }

        // Высота камеры относительно пола (ось Y)
        public double Height { get; set; }
    }

    public class SlamProcessor
    {
        private static readonly int MIN_FRAMES_FOR_SLAM = 15;
        private static readonly int MAP_SIZE = 800;
        private static readonly int MAP_SCALE = 80;
        private static readonly Random random = new Random();

        private static Mat slamMap;
        private static List<VectorOfKeyPoint> allKeypoints = new List<VectorOfKeyPoint>();
        private static List<Mat> allDescriptors = new List<Mat>();
        private static List<Tuple<double, double, double>> trajectory = new List<Tuple<double, double, double>>();
        private static int processedFramesCount = 0;
        private static bool isInitialized = false;

        private static void Initialize()
        {
            if (!isInitialized)
            {
                slamMap = new Mat(MAP_SIZE, MAP_SIZE, DepthType.Cv8U, 3);
                slamMap.SetTo(new MCvScalar(255, 255, 255));
                isInitialized = true;
            }
        }

        public static StereoFrameResponse ProcessStereoFrame(StereoFrameRequest request)
        {
            Initialize();

            // Обновление позиции робота в траектории
            trajectory.Add(new Tuple<double, double, double>(request.RobotX, request.RobotY, request.RobotAngleY));

            // Преобразование входных изображений
            Mat leftImage = ConvertByteArrayToMat(request.LeftImageData);
            Mat rightImage = ConvertByteArrayToMat(request.RightImageData);

            // Предобработка изображений
            Mat leftGray = PreprocessImage(leftImage);
            Mat rightGray = PreprocessImage(rightImage);

            // Извлечение ключевых точек и дескрипторов
            var leftResult = ExtractFeatures(leftGray);
            var rightResult = ExtractFeatures(rightGray);

            VectorOfKeyPoint leftKeypoints = leftResult.Item1;
            Mat leftDescriptors = leftResult.Item2;

            VectorOfKeyPoint rightKeypoints = rightResult.Item1;
            Mat rightDescriptors = rightResult.Item2;

            // Сохранение ключевых точек и дескрипторов
            allKeypoints.Add(leftKeypoints);
            allDescriptors.Add(leftDescriptors);

            // Визуализация ключевых точек
            Mat leftVisual = leftImage.Clone();
            Mat rightVisual = rightImage.Clone();

            Features2DToolbox.DrawKeypoints(leftImage, leftKeypoints, leftVisual, new Bgr(0, 0, 255), Features2DToolbox.KeypointDrawType.DrawRichKeypoints);
            Features2DToolbox.DrawKeypoints(rightImage, rightKeypoints, rightVisual, new Bgr(0, 0, 255), Features2DToolbox.KeypointDrawType.DrawRichKeypoints);

            // Обновление SLAM карты
            UpdateSlamMap();

            // Расчет следующей позиции для робота
            var (nextX, nextY, nextAngleY) = CalculateNextPosition(request.RobotX, request.RobotY, request.RobotAngleY, request.CartographerSpeed);

            // Проверка завершения SLAM
            processedFramesCount++;
            bool isComplete = IsSlammingComplete();

            // Подготовка ответа
            var response = new StereoFrameResponse
            {
                ProcessedLeftImage = ConvertMatToByteArray(leftVisual),
                ProcessedRightImage = ConvertMatToByteArray(rightVisual),
                SlamMapImage = ConvertMatToByteArray(slamMap),
                NextX = (float)nextX,
                NextY = (float)nextY,
                NextAngleY = (float)nextAngleY,
                NextAngleX = 0,
                NextAngleZ = 0,
                IsSlammingComplete = isComplete
            };

            return response;
        }

        private static Mat PreprocessImage(Mat image)
        {
            Mat gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            // Гистограммная нормализация
            Mat normalized = new Mat();
            CvInvoke.EqualizeHist(gray, normalized);

            return normalized;
        }

        private static Tuple<VectorOfKeyPoint, Mat> ExtractFeatures(Mat image)
        {
            var detector = new SIFT();
            var keypoints = new VectorOfKeyPoint();
            Mat descriptors = new Mat();

            detector.DetectAndCompute(image, null, keypoints, descriptors, false);

            return new Tuple<VectorOfKeyPoint, Mat>(keypoints, descriptors);
        }

        private static void UpdateSlamMap()
        {
            // Очистка карты
            slamMap.SetTo(new MCvScalar(255, 255, 255));

            // Рисование сетки
            DrawGrid();

            // Рисование осей координат
            DrawAxes();

            // Рисование траектории
            DrawTrajectory();

            // Рисование ключевых точек
            DrawKeypoints();
        }

        private static void DrawGrid()
        {
            int cellSize = MAP_SCALE;
            for (int i = 0; i < MAP_SIZE; i += cellSize)
            {
                CvInvoke.Line(slamMap,
                    new System.Drawing.Point(i, 0),
                    new System.Drawing.Point(i, MAP_SIZE),
                    new MCvScalar(230, 230, 230), 1);

                CvInvoke.Line(slamMap,
                    new System.Drawing.Point(0, i),
                    new System.Drawing.Point(MAP_SIZE, i),
                    new MCvScalar(230, 230, 230), 1);
            }
        }

        private static void DrawAxes()
        {
            // Центр карты
            int centerX = MAP_SIZE / 2;
            int centerY = MAP_SIZE / 2;

            // Ось X (красная)
            CvInvoke.Line(slamMap,
                new System.Drawing.Point(centerX, centerY),
                new System.Drawing.Point(centerX + MAP_SCALE, centerY),
                new MCvScalar(0, 0, 255), 2);

            // Ось Y (зеленая)
            CvInvoke.Line(slamMap,
                new System.Drawing.Point(centerX, centerY),
                new System.Drawing.Point(centerX, centerY - MAP_SCALE),
                new MCvScalar(0, 255, 0), 2);
        }

        private static void DrawTrajectory()
        {
            if (trajectory.Count < 2) return;

            int centerX = MAP_SIZE / 2;
            int centerY = MAP_SIZE / 2;

            for (int i = 1; i < trajectory.Count; i++)
            {
                var prev = trajectory[i - 1];
                var curr = trajectory[i];

                int prevX = centerX + (int)(prev.Item1 * MAP_SCALE);
                int prevY = centerY - (int)(prev.Item2 * MAP_SCALE);

                int currX = centerX + (int)(curr.Item1 * MAP_SCALE);
                int currY = centerY - (int)(curr.Item2 * MAP_SCALE);

                // Проверка границ карты
                if (IsInBounds(prevX, prevY) && IsInBounds(currX, currY))
                {
                    // Рисуем линию траектории (синяя)
                    CvInvoke.Line(slamMap,
                        new System.Drawing.Point(prevX, prevY),
                        new System.Drawing.Point(currX, currY),
                        new MCvScalar(255, 0, 0), 2);

                    // Рисуем направление (красная стрелка)
                    double angle = curr.Item3;
                    int arrowLength = 15;
                    int arrowX = (int)(currX + arrowLength * Math.Sin(angle));
                    int arrowY = (int)(currY - arrowLength * Math.Cos(angle));

                    if (IsInBounds(arrowX, arrowY))
                    {
                        CvInvoke.ArrowedLine(slamMap,
                            new System.Drawing.Point(currX, currY),
                            new System.Drawing.Point(arrowX, arrowY),
                            new MCvScalar(0, 0, 255), 2);
                    }
                }
            }
        }

        private static void DrawKeypoints()
        {
            int centerX = MAP_SIZE / 2;
            int centerY = MAP_SIZE / 2;

            for (int i = 0; i < allKeypoints.Count; i++)
            {
                if (i >= trajectory.Count) continue;

                var currPos = trajectory[i];
                var keypoints = allKeypoints[i];

                int robotX = centerX + (int)(currPos.Item1 * MAP_SCALE);
                int robotY = centerY - (int)(currPos.Item2 * MAP_SCALE);

                // Получаем массив ключевых точек из VectorOfKeyPoint
                MKeyPoint[] kpArray = keypoints.ToArray();

                foreach (var kp in kpArray)
                {
                    // Используем размер ключевой точки для имитации глубины
                    int depth = (int)(5 + kp.Size);

                    // Случайный цвет для ключевой точки в зависимости от глубины
                    byte r = (byte)(50 + (depth * 10) % 150);
                    byte g = (byte)(50 + (depth * 15) % 150);
                    byte b = (byte)(50 + (depth * 20) % 150);

                    // Относительные координаты точки (с учетом угла)
                    double angle = currPos.Item3;
                    double relX = Math.Cos(angle) * kp.Point.X - Math.Sin(angle) * kp.Point.Y;
                    double relY = Math.Sin(angle) * kp.Point.X + Math.Cos(angle) * kp.Point.Y;

                    int pointX = robotX + (int)(relX / 10);
                    int pointY = robotY - (int)(relY / 10);

                    if (IsInBounds(pointX, pointY))
                    {
                        CvInvoke.Circle(slamMap,
                            new System.Drawing.Point(pointX, pointY),
                            2, new MCvScalar(b, g, r), -1);
                    }
                }
            }
        }

        private static bool IsInBounds(int x, int y)
        {
            return x >= 0 && x < MAP_SIZE && y >= 0 && y < MAP_SIZE;
        }

        private static Tuple<double, double, double> CalculateNextPosition(double currentX, double currentY, double currentAngle, double speed)
        {
            // Максимальное смещение для движения
            double maxMovement = speed * 0.2;
            double minMovement = speed * 0.05;

            // Случайное смещение вперед
            double distance = minMovement + random.NextDouble() * (maxMovement - minMovement);

            // Случайное изменение угла
            double angleChange = (random.NextDouble() - 0.5) * Math.PI / 4;
            double newAngle = currentAngle + angleChange;

            // Сохраняем угол в диапазоне [0, 2π]
            newAngle = (newAngle + 2 * Math.PI) % (2 * Math.PI);

            // Расчет новых координат с учетом угла
            double newX = currentX + distance * Math.Sin(newAngle);
            double newY = currentY + distance * Math.Cos(newAngle);

            // Проверка на выход за границы сцены (условно ±5 единиц)
            const double SCENE_BOUNDARY = 5.0;

            if (newX > SCENE_BOUNDARY) { newX = SCENE_BOUNDARY; newAngle = -newAngle; }
            if (newX < -SCENE_BOUNDARY) { newX = -SCENE_BOUNDARY; newAngle = -newAngle; }
            if (newY > SCENE_BOUNDARY) { newY = SCENE_BOUNDARY; newAngle = Math.PI - newAngle; }
            if (newY < -SCENE_BOUNDARY) { newY = -SCENE_BOUNDARY; newAngle = Math.PI - newAngle; }

            return new Tuple<double, double, double>(newX, newY, newAngle);
        }

        private static bool IsSlammingComplete()
        {
            if (processedFramesCount < MIN_FRAMES_FOR_SLAM)
                return false;

            return HasGoodCoverage();
        }

        private static bool HasGoodCoverage()
        {
            // Минимальная площадь покрытия: 10x10 условных единиц
            if (trajectory.Count < 10) return false;

            double minX = trajectory.Min(p => p.Item1);
            double maxX = trajectory.Max(p => p.Item1);
            double minY = trajectory.Min(p => p.Item2);
            double maxY = trajectory.Max(p => p.Item2);

            double width = maxX - minX;
            double height = maxY - minY;

            return (width >= 1.0 && height >= 1.0);
        }

        private static Mat ConvertByteArrayToMat(byte[] imageData)
        {
            if (imageData == null || imageData.Length == 0)
            {
                throw new ArgumentException("Image data cannot be null or empty", nameof(imageData));
            }

            Mat result = new Mat();

            try
            {
                CvInvoke.Imdecode(imageData, ImreadModes.Color, result);

                if (result.IsEmpty)
                {
                    throw new ArgumentException("Failed to decode image - possibly corrupted or unsupported format");
                }

                return result;
            }
            catch (Exception ex)
            {
                result?.Dispose();
                throw new ArgumentException("Failed to convert byte array to Mat", nameof(imageData), ex);
            }
        }

        private static byte[] ConvertMatToByteArray(Mat image)
        {
            if (image == null || image.IsEmpty)
            {
                throw new ArgumentException("Image cannot be null or empty", nameof(image));
            }

            using (VectorOfByte vb = new VectorOfByte())
            {
                CvInvoke.Imencode(".png", image, vb);
                return vb.ToArray();
            }
        }
    }
}