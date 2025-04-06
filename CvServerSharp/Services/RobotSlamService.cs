using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Runtime.InteropServices;

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
        private List<System.Drawing.PointF> _slamPoints = new List<System.Drawing.PointF>();

        public SlamProcessor(ILogger<SlamProcessor> logger)
        {
            _logger = logger;
        }

        public StereoFrameResponse ProcessStereoFrame(StereoFrameRequest request)
        {
            try
            {
                _logger.LogInformation("[ProcessStereoFrame] Получение новых стерео кадров");

                if (request == null ||
                    request.LeftImageData == null || request.LeftImageData.Length == 0 ||
                    request.RightImageData == null || request.RightImageData.Length == 0)
                {
                    throw new ArgumentException("Отсутствуют данные изображений");
                }

                // Обновление позиции и ориентации робота из запроса.
                _robotX = request.RobotX;
                _robotY = request.RobotY;
                double robotAngleX = request.RobotAngleX;
                double robotAngleY = request.RobotAngleY;
                double robotAngleZ = request.RobotAngleZ;
                double cartographerSpeed = request.CartographerSpeed;

                // Декодирование левого и правого изображений.
                using Mat leftImgMat = DecodeImage(request.LeftImageData);
                using Mat rightImgMat = DecodeImage(request.RightImageData);

                // Обработка изображений для извлечения ключевых точек и дескрипторов.
                ImageProcessing.ProcessImage(leftImgMat, out VectorOfKeyPoint leftKeypoints, out Mat leftDescriptors);
                ImageProcessing.ProcessImage(rightImgMat, out VectorOfKeyPoint rightKeypoints, out Mat rightDescriptors);

                double dx = 0.0, dy = 0.0, angle = 0.0;
                List<System.Drawing.PointF> newPoints = new List<System.Drawing.PointF>();

                // Если есть предыдущие изображения, пытаемся оценить смещение и извлечь новые точки.
                if (_prevLeftImage != null && _prevRightImage != null)
                {
                    try
                    {
                        var stereoMatches = ImageProcessing.MatchDescriptors(leftDescriptors, rightDescriptors);
                        newPoints = ProcessStereoMatches(leftKeypoints, rightKeypoints, stereoMatches,
                                                        request.LeftCameraInfo, request.RightCameraInfo);

                        // Вычисляем смещение между предыдущим и текущим кадрами.
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

                // Добавление новых точек для формирования SLAM карты.
                _slamPoints.AddRange(newPoints);

                // Создание обработанных изображений для визуализации.
                Mat processedLeftImage = CreateProcessedImage(leftImgMat, leftKeypoints);
                Mat processedRightImage = CreateProcessedImage(rightImgMat, rightKeypoints);
                Mat slamMapImage = CreateSlamMapImage();

                // Расчет следующей позиции и ориентации робота с учетом положения камеры.
                (float nextX, float nextY, float nextAngleX, float nextAngleY, float nextAngleZ) =
                    DetermineNextPosition(dx, dy, angle, robotAngleX, robotAngleY, robotAngleZ, cartographerSpeed, request.LeftCameraInfo);

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

                // Обновление предыдущих кадров и параметров для следующих итераций.
                UpdatePreviousFrames(leftImgMat, rightImgMat, leftKeypoints, rightKeypoints, leftDescriptors, rightDescriptors);

                // Освобождение ресурсов временных изображений
                processedLeftImage.Dispose();
                processedRightImage.Dispose();
                slamMapImage.Dispose();

                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Необработанная ошибка в ProcessStereoFrame");
                throw new Exception($"Ошибка обработки: {ex.Message}");
            }
        }

        // Метод, учитывающий параметры камеры при расчете следующей позиции
        private (float, float, float, float, float) DetermineNextPosition(double dx, double dy, double angle,
            double robotAngleX, double robotAngleY, double robotAngleZ, double cartographerSpeed, CameraInfo camera)
        {
            // Учитываем смещение камеры для корректировки расчетов движения
            double adjustedDx = dx;
            double adjustedDy = dy;

            // Корректировка dx и dy с учетом смещения камеры по X и Z
            // Преобразуем смещения, наблюдаемые камерой, в смещения центра робота
            if (camera != null)
            {
                // Угол поворота робота в радианах
                double robotAngleZRad = robotAngleZ * Math.PI / 180.0;

                // Учет смещения камеры при определении движения робота
                // Проецирование смещений камеры на оси робота
                double cameraCosComponent = camera.OffsetX * Math.Cos(robotAngleZRad) - camera.OffsetZ * Math.Sin(robotAngleZRad);
                double cameraSinComponent = camera.OffsetX * Math.Sin(robotAngleZRad) + camera.OffsetZ * Math.Cos(robotAngleZRad);

                // Корректировка смещений с учетом положения камеры
                adjustedDx = dx + cameraCosComponent;
                adjustedDy = dy + cameraSinComponent;
            }

            // Вычисление нового положения робота
            double stepSize = cartographerSpeed;

            // Применение смещения с учетом скорости и направления
            double nextAngleZ = (robotAngleZ + angle) % 360;
            double nextAngleZRad = nextAngleZ * Math.PI / 180.0;

            double nextX = _robotX + adjustedDx * stepSize * Math.Cos(nextAngleZRad);
            double nextY = _robotY + adjustedDy * stepSize * Math.Sin(nextAngleZRad);

            // Возвращаем результаты в виде float для соответствия типам в StereoFrameResponse
            return ((float)nextX, (float)nextY, (float)robotAngleX, (float)robotAngleY, (float)nextAngleZ);
        }

        // Определяет, завершено ли картографирование. 
        // Логика может быть расширена в соответствии с конкретными требованиями.
        private bool IsSlammingComplete()
        {
            // Пример простого условия: если собрано определенное количество точек
            const int requiredPointCount = 1000;
            return _slamPoints.Count >= requiredPointCount;
        }

        // Создание карты SLAM на основе собранных точек.
        private Mat CreateSlamMapImage()
        {
            // Создаем пустое изображение для карты
            Mat slamMap = new Mat(600, 800, Emgu.CV.CvEnum.DepthType.Cv8U, 3);
            slamMap.SetTo(new MCvScalar(255, 255, 255)); // Белый фон

            // Центр карты
            int centerX = slamMap.Width / 2;
            int centerY = slamMap.Height / 2;

            // Рисуем точки на карте
            foreach (var point in _slamPoints)
            {
                // Масштабирование и смещение точек
                int x = centerX + (int)(point.X * 5); // Увеличение масштаба для лучшей видимости
                int y = centerY + (int)(point.Y * 5);

                // Проверка границ изображения
                if (x >= 0 && x < slamMap.Width && y >= 0 && y < slamMap.Height)
                {
                    // Рисуем точку на карте как маленький круг
                    CvInvoke.Circle(slamMap, new System.Drawing.Point(x, y), 2, new MCvScalar(0, 0, 255), -1);
                }
            }

            // Рисуем текущую позицию робота
            int robotX = centerX + (int)(_robotX * 5);
            int robotY = centerY + (int)(_robotY * 5);
            if (robotX >= 0 && robotX < slamMap.Width && robotY >= 0 && robotY < slamMap.Height)
            {
                CvInvoke.Circle(slamMap, new System.Drawing.Point(robotX, robotY), 5, new MCvScalar(0, 255, 0), -1);
            }

            return slamMap;
        }

        // Обработка стереоизображений для получения 3D-точек.
        private List<System.Drawing.PointF> ProcessStereoMatches(
            VectorOfKeyPoint leftKeypoints,
            VectorOfKeyPoint rightKeypoints,
            VectorOfVectorOfDMatch matches,
            CameraInfo leftCameraInfo,
            CameraInfo rightCameraInfo)
        {
            List<System.Drawing.PointF> points3D = new List<System.Drawing.PointF>();

            if (matches == null || matches.Size == 0)
                return points3D;

            // Фильтрация совпадений с использованием теста отношения Лоу
            var goodMatches = new List<MDMatch>();
            var matchesArray = matches.ToArrayOfArray();

            for (int i = 0; i < matchesArray.Length; i++)
            {
                if (matchesArray[i].Length >= 2 &&
                    matchesArray[i][0].Distance < 0.7 * matchesArray[i][1].Distance)
                {
                    goodMatches.Add(matchesArray[i][0]);
                }
            }

            // Расчет 3D точек из стерео соответствий
            foreach (var match in goodMatches)
            {
                var leftPoint = leftKeypoints[match.QueryIdx].Point;
                var rightPoint = rightKeypoints[match.TrainIdx].Point;

                // Дисперсия (разница в горизонтальных координатах)
                float disparity = leftPoint.X - rightPoint.X;

                // Избегаем деления на ноль или на очень маленькие значения
                if (Math.Abs(disparity) < 0.1)
                    continue;

                // Расчет 3D координат с использованием параметров камеры
                // (упрощенная версия, предполагает ректифицированные изображения)
                float f = (float)(leftCameraInfo?.FocalLength ?? 500); // Используем фокусное расстояние или значение по умолчанию
                float b = (float)(Math.Abs(leftCameraInfo.OffsetX - rightCameraInfo.OffsetX)); // Базовая линия

                float Z = f * b / disparity; // Глубина
                float X = (leftPoint.X - (float)(leftCameraInfo?.PrincipalPointX ?? 0)) * Z / f; // Горизонтальная координата
                float Y = (leftPoint.Y - (float)(leftCameraInfo?.PrincipalPointY ?? 0)) * Z / f; // Вертикальная координата

                // Проверка разумности значений глубины
                if (Z > 0 && Z < 100) // Ограничение на глубину для отсечения шумных результатов
                {
                    // Преобразуем в 2D точку (план сверху) для карты SLAM
                    points3D.Add(new System.Drawing.PointF(X, Z));
                }
            }

            return points3D;
        }

        // Создание визуализации обработанного изображения.
        private Mat CreateProcessedImage(Mat originalImage, VectorOfKeyPoint keypoints)
        {
            Mat output = new Mat();
            CvInvoke.CvtColor(originalImage, output, Emgu.CV.CvEnum.ColorConversion.Bgr2Bgra);

            // Рисуем ключевые точки на изображении
            Features2DToolbox.DrawKeypoints(output, keypoints, output, new Bgr(0, 255, 0));

            return output;
        }

        // Обновляет предыдущие кадры для следующей итерации.
        private void UpdatePreviousFrames(Mat leftImgMat, Mat rightImgMat,
                                         VectorOfKeyPoint leftKeypoints, VectorOfKeyPoint rightKeypoints,
                                         Mat leftDescriptors, Mat rightDescriptors)
        {
            // Освобождение ресурсов предыдущих изображений
            _prevLeftImage?.Dispose();
            _prevRightImage?.Dispose();
            _prevLeftKeypoints?.Dispose();
            _prevRightKeypoints?.Dispose();
            _prevLeftDescriptors?.Dispose();
            _prevRightDescriptors?.Dispose();

            // Клонирование текущих изображений для использования в следующей итерации
            _prevLeftImage = leftImgMat.Clone();
            _prevRightImage = rightImgMat.Clone();

            // Создание новых экземпляров для ключевых точек и дескрипторов
            _prevLeftKeypoints = new VectorOfKeyPoint();
            for (int i = 0; i < leftKeypoints.Size; i++)
                _prevLeftKeypoints.Push(new MKeyPoint[] { leftKeypoints[i] });

            _prevRightKeypoints = new VectorOfKeyPoint();
            for (int i = 0; i < rightKeypoints.Size; i++)
                _prevRightKeypoints.Push(new MKeyPoint[] { rightKeypoints[i] });

            _prevLeftDescriptors = leftDescriptors.Clone();
            _prevRightDescriptors = rightDescriptors.Clone();
        }

        // Декодирование массива байтов в изображение.
        private Mat DecodeImage(byte[] imageData)
        {
            Mat result = new Mat();
            CvInvoke.Imdecode(imageData, Emgu.CV.CvEnum.ImreadModes.Color, result);
            return result;
        }

        // Кодирование изображения в массив байтов.
        private byte[] EncodeImage(Mat image)
        {
            if (image == null || image.IsEmpty)
            {
                throw new ArgumentException("Изображение пустое или равно null");
            }

            using VectorOfByte vectorOfByte = new VectorOfByte();
            CvInvoke.Imencode(".png", image, vectorOfByte);
            return vectorOfByte.ToArray();
        }
    }
}