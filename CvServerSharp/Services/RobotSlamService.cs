using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Xml.Linq;

namespace RobotSlamServer
{
    public class StereoFrameRequest
    {
        public double RobotX { get; set; }
        public double RobotY { get; set; }
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

    // Интерфейс логгера, который можно заменить на другой механизм логирования
    public interface ILogger
    {
        void LogInformation(string message);
        void LogError(Exception ex, string message, params object[] args);
    }

    public static class ImageProcessing
    {
        // Метод обработки изображения для извлечения ключевых точек и дескрипторов.
        public static void ProcessImage(Mat image, out VectorOfKeyPoint keypoints, out Mat descriptors)
        {
            keypoints = new VectorOfKeyPoint();
            descriptors = new Mat();
            // Используем детектор ORB для извлечения ключевых точек и дескрипторов.
            using (var orb = new ORB())
            {
                orb.DetectAndCompute(image, null, keypoints, descriptors, false);
            }
        }

        // Метод для сопоставления дескрипторов с использованием BFMatcher
        public static VectorOfVectorOfDMatch MatchDescriptors(Mat descriptors1, Mat descriptors2)
        {
            var matcher = new BFMatcher(DistanceType.Hamming);
            var matches = new VectorOfVectorOfDMatch();

            if (!descriptors1.IsEmpty && !descriptors2.IsEmpty)
            {
                matcher.KnnMatch(descriptors1, descriptors2, matches, 2);
            }

            return matches;
        }

        // Метод для оценки матрицы преобразования между двумя наборами ключевых точек
        public static Mat EstimateTransform(VectorOfKeyPoint prevKeypoints, VectorOfKeyPoint currentKeypoints, VectorOfVectorOfDMatch matches)
        {
            // Фильтрация совпадений с использованием теста отношения Лоу
            List<MDMatch[]> goodMatches = FilterMatches(matches);

            if (goodMatches.Count < 4)
            {
                return null;
            }

            // Извлечение соответствующих точек
            var prevPoints = new List<System.Drawing.PointF>();
            var currentPoints = new List<System.Drawing.PointF>();

            foreach (var match in goodMatches)
            {
                prevPoints.Add(prevKeypoints[match[0].QueryIdx].Point);
                currentPoints.Add(currentKeypoints[match[0].TrainIdx].Point);
            }

            // Оценка матрицы аффинного преобразования
            Mat transformMatrix = CvInvoke.EstimateAffine2D(
                ToPointMatrix(prevPoints),
                ToPointMatrix(currentPoints),
                method: Emgu.CV.CvEnum.RobustEstimationAlgorithm.Ransac,
                ransacReprojThreshold: 5.0,
                maxIters: 2000,
                confidence: 0.99,
                refineIters: 10  // Добавлен параметр для количества итераций уточнения
            );

            return transformMatrix;
        }

        // Преобразование списка точек в Mat для использования в CvInvoke.EstimateAffine2D
        private static Mat ToPointMatrix(List<System.Drawing.PointF> points)
        {
            // Создаем матрицу размером Nx1, где каждый элемент - точка с 2 координатами (x,y)
            Mat result = new Mat(points.Count, 1, DepthType.Cv32F, 2);

            // Заполняем матрицу точками
            for (int i = 0; i < points.Count; i++)
            {
                // Создаем массив для координат точки
                float[] pointData = new float[] { points[i].X, points[i].Y };

                // Используем метод SetTo для установки значений
                using (Mat pointMat = new Mat(1, 1, DepthType.Cv32F, 2, Marshal.UnsafeAddrOfPinnedArrayElement(pointData, 0), 8))
                {
                    // Копируем данные в указанную строку результирующей матрицы
                    using (Mat subMat = result.Row(i))
                    {
                        pointMat.CopyTo(subMat);
                    }
                }
            }
            return result;
        }

        // Фильтрация совпадений с использованием теста отношения Лоу
        private static List<MDMatch[]> FilterMatches(VectorOfVectorOfDMatch matches)
        {
            var goodMatches = new List<MDMatch[]>();
            var matchesArray = matches.ToArrayOfArray();

            for (int i = 0; i < matchesArray.Length; i++)
            {
                if (matchesArray[i].Length >= 2)
                {
                    if (matchesArray[i][0].Distance < 0.7 * matchesArray[i][1].Distance)
                    {
                        goodMatches.Add(matchesArray[i]);
                    }
                }
            }

            return goodMatches;
        }

        // Вычисление смещения и поворота на основе матрицы преобразования
        public static (double dx, double dy, double angle) ComputeOffsetAndRotation(Mat M)
        {
            if (M == null || M.IsEmpty)
            {
                return (0, 0, 0);
            }

            double m00;
            double m01;
            double m10;
            double m11;
            double dx;
            double dy;

            using (Matrix<double> mat = new Matrix<double>(M.Rows, M.Cols, M.NumberOfChannels))
            {
                M.CopyTo(mat);
                dx = mat[0, 2];
                dy = mat[1, 2];
                m00 = mat[0, 0];
                m01 = mat[0, 1];
                m10 = mat[1, 0];
                m11 = mat[1, 1];
            }

            // Вычисление угла поворота (в радианах, затем в градусах)
            double angle = Math.Atan2(m10, m00);
            angle = angle * 180 / Math.PI;

            return (dx, dy, angle);
        }
    }

    public class SlamProcessor
    {
        private readonly ILogger<SlamProcessor> _logger;
        private double _robotX, _robotY;
        private Mat _prevLeftImage, _prevRightImage;
        private VectorOfKeyPoint _prevLeftKeypoints, _prevRightKeypoints;
        private Mat _prevLeftDescriptors, _prevRightDescriptors;
        private List<PointF> _slamPoints = new List<PointF>();
        private const string XmlFilePath = "SlamProcessorData.xml";

        public SlamProcessor(ILogger<SlamProcessor> logger)
        {
            _logger = logger;
            LoadStateFromXml();
        }

        public StereoFrameResponse ProcessStereoFrame(StereoFrameRequest request)
        {
            try
            {
                if (request == null ||
                    request.LeftImageData == null || request.LeftImageData.Length == 0 ||
                    request.RightImageData == null || request.RightImageData.Length == 0)
                {
                    throw new ArgumentException("Отсутствуют данные изображений");
                }
                _robotX = request.RobotX;
                _robotY = request.RobotY;
                double robotAngleX = request.RobotAngleX;
                double robotAngleY = request.RobotAngleY;
                double robotAngleZ = request.RobotAngleZ;
                double cartographerSpeed = request.CartographerSpeed;
                using Mat leftImgMat = DecodeImage(request.LeftImageData);
                using Mat rightImgMat = DecodeImage(request.RightImageData);
                ImageProcessing.ProcessImage(leftImgMat, out VectorOfKeyPoint leftKeypoints, out Mat leftDescriptors);
                ImageProcessing.ProcessImage(rightImgMat, out VectorOfKeyPoint rightKeypoints, out Mat rightDescriptors);
                double dx = 0.0, dy = 0.0, angle = 0.0;
                List<PointF> newPoints = new List<PointF>();
                if (_prevLeftImage != null && _prevRightImage != null)
                {
                    try
                    {
                        var stereoMatches = ImageProcessing.MatchDescriptors(leftDescriptors, rightDescriptors);
                        newPoints = ProcessStereoMatches(leftKeypoints, rightKeypoints, stereoMatches,
                                                         request.LeftCameraInfo, request.RightCameraInfo);
                        var leftMatches = ImageProcessing.MatchDescriptors(_prevLeftDescriptors, leftDescriptors);
                        if (leftMatches != null && leftMatches.Size >= 4)
                        {
                            Mat M = ImageProcessing.EstimateTransform(_prevLeftKeypoints, leftKeypoints, leftMatches);
                            if (M != null && !M.IsEmpty)
                            {
                                (dx, dy, angle) = ImageProcessing.ComputeOffsetAndRotation(M);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Ошибка при обработке стерео изображений: {0}", ex.Message);
                    }
                }
                _slamPoints.AddRange(newPoints);
                Mat processedLeftImage = CreateProcessedImage(leftImgMat, leftKeypoints);
                Mat processedRightImage = CreateProcessedImage(rightImgMat, rightKeypoints);
                Mat slamMapImage = CreateSlamMapImage();
                (float nextX, float nextY, float nextAngleX, float nextAngleY, float nextAngleZ) =
                    DetermineNextPosition(_robotX, _robotY, robotAngleX, robotAngleY, robotAngleZ, cartographerSpeed, request.LeftCameraInfo, request.LeftCameraInfo);
                StereoFrameResponse response = new StereoFrameResponse
                {
                    ProcessedLeftImage = EncodeImage(processedLeftImage),
                    ProcessedRightImage = EncodeImage(processedRightImage),
                    SlamMapImage = EncodeImage(slamMapImage),
                    NextX = nextX,
                    NextY = nextY,
                    NextAngleX = nextAngleX,
                    NextAngleY = nextAngleY,
                    NextAngleZ = nextAngleZ,
                    IsSlammingComplete = IsSlammingComplete()
                };
                UpdatePreviousFrames(leftImgMat, rightImgMat, leftKeypoints, rightKeypoints, leftDescriptors, rightDescriptors);
                processedLeftImage.Dispose();
                processedRightImage.Dispose();
                slamMapImage.Dispose();
                SaveStateToXml();
                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Необработанная ошибка в ProcessStereoFrame");
                throw new Exception($"Ошибка обработки: {ex.Message}");
            }
        }

        public (float, float, float, float, float) DetermineNextPosition(
            double robotX, double robotY,
            double robotAngleX, double robotAngleY, double robotAngleZ,
            double cartographerSpeed, CameraInfo leftCamera, CameraInfo rightCamera)
        {
            double adjustedDx = 0;
            double adjustedDy = 0;
            int count = 0;

            double robotAngleZRad = robotAngleZ * Math.PI / 180.0;

            // С учётом оффсетов камер
            if (leftCamera != null)
            {
                double leftDx = leftCamera.OffsetX * Math.Cos(robotAngleZRad) - leftCamera.OffsetZ * Math.Sin(robotAngleZRad);
                double leftDy = leftCamera.OffsetX * Math.Sin(robotAngleZRad) + leftCamera.OffsetZ * Math.Cos(robotAngleZRad);
                adjustedDx += leftDx;
                adjustedDy += leftDy;
                count++;
            }

            if (rightCamera != null)
            {
                double rightDx = rightCamera.OffsetX * Math.Cos(robotAngleZRad) - rightCamera.OffsetZ * Math.Sin(robotAngleZRad);
                double rightDy = rightCamera.OffsetX * Math.Sin(robotAngleZRad) + rightCamera.OffsetZ * Math.Cos(robotAngleZRad);
                adjustedDx += rightDx;
                adjustedDy += rightDy;
                count++;
            }

            if (count > 0)
            {
                adjustedDx /= count;
                adjustedDy /= count;
            }

            // Если накоплены slam-точки, вычисляем их центроид
            if (_slamPoints != null && _slamPoints.Count > 0)
            {
                double sumX = 0;
                double sumY = 0;
                foreach (var pt in _slamPoints)
                {
                    sumX += pt.X;
                    sumY += pt.Y;
                }
                double centroidX = sumX / _slamPoints.Count;
                double centroidY = sumY / _slamPoints.Count;

                // Определим вектор от текущей позиции робота до центроида slam-точек
                double vectorToCentroidX = centroidX - robotX;
                double vectorToCentroidY = centroidY - robotY;

                // Смешиваем предложенное смещение от камер и направление к центроиду slam-точек.
                // Можно выбрать разные веса, здесь используется простое среднее.
                adjustedDx = (adjustedDx + vectorToCentroidX) / 2.0;
                adjustedDy = (adjustedDy + vectorToCentroidY) / 2.0;
            }

            double nextX = robotX + adjustedDx * cartographerSpeed;
            double nextY = robotY + adjustedDy * cartographerSpeed;

            // Обновляем позицию робота, если необходимо использовать её дальше в логику
            _robotX = nextX;
            _robotY = nextY;

            // Возвращаем новое положение и необработанные углы. При необходимости можно доработать логику углов.
            return ((float)nextX, (float)nextY, (float)robotAngleX, (float)robotAngleY, (float)robotAngleZ);
        }

        private bool IsSlammingComplete()
        {
            const int requiredPointCount = 1000;
            return _slamPoints.Count >= requiredPointCount;
        }

        private Mat CreateSlamMapImage()
        {
            Mat slamMap = new Mat(600, 800, DepthType.Cv8U, 3);
            slamMap.SetTo(new MCvScalar(255, 255, 255));
            int centerX = slamMap.Width / 2;
            int centerY = slamMap.Height / 2;
            foreach (var point in _slamPoints)
            {
                int x = centerX + (int)(point.X * 5);
                int y = centerY + (int)(point.Y * 5);
                if (x >= 0 && x < slamMap.Width && y >= 0 && y < slamMap.Height)
                {
                    CvInvoke.Circle(slamMap, new System.Drawing.Point(x, y), 2, new MCvScalar(0, 0, 255), -1);
                }
            }
            int robotX = centerX + (int)(_robotX * 5);
            int robotY = centerY + (int)(_robotY * 5);
            if (robotX >= 0 && robotX < slamMap.Width && robotY >= 0 && robotY < slamMap.Height)
            {
                CvInvoke.Circle(slamMap, new System.Drawing.Point(robotX, robotY), 5, new MCvScalar(0, 255, 0), -1);
            }
            return slamMap;
        }

        private List<PointF> ProcessStereoMatches(
            VectorOfKeyPoint leftKeypoints,
            VectorOfKeyPoint rightKeypoints,
            VectorOfVectorOfDMatch matches,
            CameraInfo leftCameraInfo,
            CameraInfo rightCameraInfo)
        {
            List<PointF> points3D = new List<PointF>();

            // Проверка на null и пустые данные
            if (matches == null || matches.Size == 0 || leftKeypoints == null || rightKeypoints == null ||
                leftCameraInfo == null || rightCameraInfo == null)
                return points3D;

            var goodMatches = new List<MDMatch>();
            var matchesArray = matches.ToArrayOfArray();

            // Исправлена проверка длины массива
            for (int i = 0; i < matchesArray.Length; i++)
            {
                // Проверяем, что текущий массив имеет как минимум 2 элемента
                if (matchesArray[i].Length >= 2 && matchesArray[i][0].Distance < 0.7 * matchesArray[i][1].Distance)
                {
                    goodMatches.Add(matchesArray[i][0]);
                }
            }

            foreach (var match in goodMatches)
            {
                // Проверка индексов перед доступом
                if (match.QueryIdx < 0 || match.QueryIdx >= leftKeypoints.Size ||
                    match.TrainIdx < 0 || match.TrainIdx >= rightKeypoints.Size)
                    continue;

                MKeyPoint leftPoint = leftKeypoints[match.QueryIdx];
                MKeyPoint rightPoint = rightKeypoints[match.TrainIdx];

                float disparity = leftPoint.Point.X - rightPoint.Point.X;

                // Изменено условие проверки диспаритета
                if (disparity <= 0 || Math.Abs(disparity) < 0.1)
                    continue;

                // Убраны нулевые проверки с использованием оператора ??
                float f = (float)leftCameraInfo.FocalLength;
                float b = (float)(Math.Abs(leftCameraInfo.OffsetX - rightCameraInfo.OffsetX));

                // Проверка на нулевой дисперитет для избежания деления на ноль
                if (disparity == 0)
                    continue;

                float Z = f * b / disparity;
                float X = (leftPoint.Point.X - (float)leftCameraInfo.PrincipalPointX) * Z / f;
                float Y = (leftPoint.Point.Y - (float)leftCameraInfo.PrincipalPointY) * Z / f;

                // Добавлена третья координата Y в результат
                if (Z > 0 && Z < 100)
                {
                    points3D.Add(new PointF(X, Y)); // Возвращаем X и Y, а не X и Z
                }
            }

            return points3D;
        }
        private Mat CreateProcessedImage(Mat originalImage, VectorOfKeyPoint keypoints)
        {
            Mat output = new Mat();
            CvInvoke.CvtColor(originalImage, output, ColorConversion.Bgr2Bgra);
            Features2DToolbox.DrawKeypoints(output, keypoints, output, new Bgr(0, 255, 0));
            return output;
        }

        private void UpdatePreviousFrames(Mat leftImgMat, Mat rightImgMat,
                                          VectorOfKeyPoint leftKeypoints, VectorOfKeyPoint rightKeypoints,
                                          Mat leftDescriptors, Mat rightDescriptors)
        {
            _prevLeftImage?.Dispose();
            _prevRightImage?.Dispose();
            _prevLeftKeypoints?.Dispose();
            _prevRightKeypoints?.Dispose();
            _prevLeftDescriptors?.Dispose();
            _prevRightDescriptors?.Dispose();
            _prevLeftImage = leftImgMat.Clone();
            _prevRightImage = rightImgMat.Clone();
            _prevLeftKeypoints = new VectorOfKeyPoint();
            foreach (var kp in leftKeypoints.ToArray())
                _prevLeftKeypoints.Push(new MKeyPoint[] { kp });
            _prevRightKeypoints = new VectorOfKeyPoint();
            foreach (var kp in rightKeypoints.ToArray())
                _prevRightKeypoints.Push(new MKeyPoint[] { kp });
            _prevLeftDescriptors = leftDescriptors.Clone();
            _prevRightDescriptors = rightDescriptors.Clone();
        }

        private Mat DecodeImage(byte[] imageData)
        {
            Mat result = new Mat();
            CvInvoke.Imdecode(imageData, ImreadModes.Color, result);
            return result;
        }

        private byte[] EncodeImage(Mat image)
        {
            if (image == null || image.IsEmpty)
                throw new ArgumentException("Изображение пустое или равно null");
            using VectorOfByte vectorOfByte = new VectorOfByte();
            CvInvoke.Imencode(".png", image, vectorOfByte);
            return vectorOfByte.ToArray();
        }

        private string EncodeMatToBase64(Mat mat)
        {
            if (mat == null || mat.IsEmpty)
                return null;
            byte[] data = new byte[mat.Rows * mat.Cols * mat.ElementSize];
            System.Runtime.InteropServices.Marshal.Copy(mat.DataPointer, data, 0, data.Length);
            return Convert.ToBase64String(data);
        }

        private Mat DecodeMatFromBase64(string base64, int rows, int cols, DepthType depth, int channels)
        {
            if (string.IsNullOrEmpty(base64))
                return null;
            byte[] data = Convert.FromBase64String(base64);
            Mat mat = new Mat(rows, cols, depth, channels);
            System.Runtime.InteropServices.Marshal.Copy(data, 0, mat.DataPointer, data.Length);
            return mat;
        }

        private void SaveStateToXml()
        {
            XElement xml = new XElement("SlamProcessorState",
                new XElement("PrevLeftImage", Convert.ToBase64String(EncodeImage(_prevLeftImage))),
                new XElement("PrevRightImage", Convert.ToBase64String(EncodeImage(_prevRightImage))),
                new XElement("PrevLeftKeypoints",
                    new XElement("Items",
                        _prevLeftKeypoints != null ? _prevLeftKeypoints.ToArray().Select(kp =>
                            new XElement("Keypoint",
                                new XAttribute("X", kp.Point.X),
                                new XAttribute("Y", kp.Point.Y),
                                new XAttribute("Size", kp.Size),
                                new XAttribute("Angle", kp.Angle),
                                new XAttribute("Response", kp.Response),
                                new XAttribute("Octave", kp.Octave),
                                new XAttribute("ClassId", kp.ClassId)
                            )) : null
                    )
                ),
                new XElement("PrevRightKeypoints",
                    new XElement("Items",
                        _prevRightKeypoints != null ? _prevRightKeypoints.ToArray().Select(kp =>
                            new XElement("Keypoint",
                                new XAttribute("X", kp.Point.X),
                                new XAttribute("Y", kp.Point.Y),
                                new XAttribute("Size", kp.Size),
                                new XAttribute("Angle", kp.Angle),
                                new XAttribute("Response", kp.Response),
                                new XAttribute("Octave", kp.Octave),
                                new XAttribute("ClassId", kp.ClassId)
                            )) : null
                    )
                ),
                new XElement("PrevLeftDescriptors",
                    new XElement("Rows", _prevLeftDescriptors?.Rows ?? 0),
                    new XElement("Cols", _prevLeftDescriptors?.Cols ?? 0),
                    new XElement("Depth", _prevLeftDescriptors != null ? (int)_prevLeftDescriptors.Depth : 0),
                    new XElement("Channels", _prevLeftDescriptors?.NumberOfChannels ?? 0),
                    new XElement("Data", EncodeMatToBase64(_prevLeftDescriptors))
                ),
                new XElement("PrevRightDescriptors",
                    new XElement("Rows", _prevRightDescriptors?.Rows ?? 0),
                    new XElement("Cols", _prevRightDescriptors?.Cols ?? 0),
                    new XElement("Depth", _prevRightDescriptors != null ? (int)_prevRightDescriptors.Depth : 0),
                    new XElement("Channels", _prevRightDescriptors?.NumberOfChannels ?? 0),
                    new XElement("Data", EncodeMatToBase64(_prevRightDescriptors))
                ),
                new XElement("SlamPoints",
                    new XElement("Items",
                        _slamPoints.Select(p =>
                            new XElement("Point",
                                new XAttribute("X", p.X),
                                new XAttribute("Y", p.Y)
                            ))
                    )
                )
            );
            xml.Save(XmlFilePath);
        }

        private void LoadStateFromXml()
        {
            if (!File.Exists(XmlFilePath))
                return;
            XElement xml = XElement.Load(XmlFilePath);
            string leftImageStr = xml.Element("PrevLeftImage")?.Value;
            if (!string.IsNullOrEmpty(leftImageStr))
            {
                byte[] leftImageData = Convert.FromBase64String(leftImageStr);
                _prevLeftImage = DecodeImage(leftImageData);
            }
            string rightImageStr = xml.Element("PrevRightImage")?.Value;
            if (!string.IsNullOrEmpty(rightImageStr))
            {
                byte[] rightImageData = Convert.FromBase64String(rightImageStr);
                _prevRightImage = DecodeImage(rightImageData);
            }
            XElement leftKpElement = xml.Element("PrevLeftKeypoints")?.Element("Items");
            if (leftKpElement != null)
            {
                _prevLeftKeypoints = new VectorOfKeyPoint();
                foreach (var kpElem in leftKpElement.Elements("Keypoint"))
                {
                    MKeyPoint kp = new MKeyPoint
                    {
                        Point = new System.Drawing.PointF(
                            float.Parse(kpElem.Attribute("X").Value),
                            float.Parse(kpElem.Attribute("Y").Value)
                        ),
                        Size = float.Parse(kpElem.Attribute("Size").Value),
                        Angle = float.Parse(kpElem.Attribute("Angle").Value),
                        Response = float.Parse(kpElem.Attribute("Response").Value),
                        Octave = int.Parse(kpElem.Attribute("Octave").Value),
                        ClassId = int.Parse(kpElem.Attribute("ClassId").Value)
                    };
                    _prevLeftKeypoints.Push(new MKeyPoint[] { kp });
                }
            }
            XElement rightKpElement = xml.Element("PrevRightKeypoints")?.Element("Items");
            if (rightKpElement != null)
            {
                _prevRightKeypoints = new VectorOfKeyPoint();
                foreach (var kpElem in rightKpElement.Elements("Keypoint"))
                {
                    MKeyPoint kp = new MKeyPoint
                    {
                        Point = new System.Drawing.PointF(
                            float.Parse(kpElem.Attribute("X").Value),
                            float.Parse(kpElem.Attribute("Y").Value)
                        ),
                        Size = float.Parse(kpElem.Attribute("Size").Value),
                        Angle = float.Parse(kpElem.Attribute("Angle").Value),
                        Response = float.Parse(kpElem.Attribute("Response").Value),
                        Octave = int.Parse(kpElem.Attribute("Octave").Value),
                        ClassId = int.Parse(kpElem.Attribute("ClassId").Value)
                    };
                    _prevRightKeypoints.Push(new MKeyPoint[] { kp });
                }
            }
            XElement leftDescElement = xml.Element("PrevLeftDescriptors");
            if (leftDescElement != null)
            {
                int rows = int.Parse(leftDescElement.Element("Rows").Value);
                int cols = int.Parse(leftDescElement.Element("Cols").Value);
                int depth = int.Parse(leftDescElement.Element("Depth").Value);
                int channels = int.Parse(leftDescElement.Element("Channels").Value);
                string dataStr = leftDescElement.Element("Data").Value;
                _prevLeftDescriptors = DecodeMatFromBase64(dataStr, rows, cols, (DepthType)depth, channels);
            }
            XElement rightDescElement = xml.Element("PrevRightDescriptors");
            if (rightDescElement != null)
            {
                int rows = int.Parse(rightDescElement.Element("Rows").Value);
                int cols = int.Parse(rightDescElement.Element("Cols").Value);
                int depth = int.Parse(rightDescElement.Element("Depth").Value);
                int channels = int.Parse(rightDescElement.Element("Channels").Value);
                string dataStr = rightDescElement.Element("Data").Value;
                _prevRightDescriptors = DecodeMatFromBase64(dataStr, rows, cols, (DepthType)depth, channels);
            }
            XElement slamPointsElement = xml.Element("SlamPoints")?.Element("Items");
            if (slamPointsElement != null)
            {
                _slamPoints = new List<PointF>();
                foreach (var ptElem in slamPointsElement.Elements("Point"))
                {
                    float x = float.Parse(ptElem.Attribute("X").Value);
                    float y = float.Parse(ptElem.Attribute("Y").Value);
                    _slamPoints.Add(new PointF(x, y));
                }
            }
        }
    }
}