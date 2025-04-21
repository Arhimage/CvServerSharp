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

    public static class SlamProcessor
    {
        private static List<PointF> _slamPoints = new List<PointF>();
        private static List<PointF> _robotPath = new List<PointF>();
        private static PointF _currentRobotPosition = new PointF(0, 0);
        private static float _currentRobotAngleY = 0;
        private static Random _random = new Random();
        private const float Scale = 80f;
        private static SIFT _siftDetector = new SIFT();

        public static StereoFrameResponse ProcessFrame(StereoFrameRequest request)
        {
            _currentRobotPosition = new PointF((float)request.RobotX, (float)request.RobotY);
            _currentRobotAngleY = (float)request.RobotAngleY;

            if (!_robotPath.Any(p => Math.Abs(p.X - _currentRobotPosition.X) < 0.1f &&
                                   Math.Abs(p.Y - _currentRobotPosition.Y) < 0.1f))
            {
                _robotPath.Add(new PointF(_currentRobotPosition.X, _currentRobotPosition.Y));
            }

            Mat leftImage = new Mat();
            CvInvoke.Imdecode(request.LeftImageData, ImreadModes.Color, leftImage);
            Mat rightImage = new Mat();
            CvInvoke.Imdecode(request.RightImageData, ImreadModes.Color, rightImage);

            Mat processedLeft = ProcessImage(leftImage);
            Mat processedRight = ProcessImage(rightImage);

            UpdateSlamMap();

            return new StereoFrameResponse
            {
                ProcessedLeftImage = processedLeft.ToImage<Bgr, byte>().ToJpegData(),
                ProcessedRightImage = processedRight.ToImage<Bgr, byte>().ToJpegData(),
                SlamMapImage = GenerateSlamMapImage(),
                NextX = GetNextX(),
                NextY = GetNextY(),
                NextAngleY = _currentRobotAngleY + (float)(_random.NextDouble() * 0.2 - 0.1),
                IsSlammingComplete = false
            };
        }

        private static Mat ProcessImage(Mat image)
        {
            if (image.IsEmpty)
                return new Mat();

            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayImage, grayImage);

            VectorOfKeyPoint keyPoints = new VectorOfKeyPoint();
            using (Mat descriptors = new Mat())
            {
                _siftDetector.DetectAndCompute(grayImage, null, keyPoints, descriptors, false);

                Mat result = image.Clone();
                if (keyPoints.Size > 0)
                {
                    Features2DToolbox.DrawKeypoints(
                        image,
                        keyPoints,
                        result,
                        new Bgr(Color.Red),
                        Features2DToolbox.KeypointDrawType.Default);

                    UpdateSlamPoints(keyPoints);
                }
                return result;
            }
        }

        private static void UpdateSlamPoints(VectorOfKeyPoint keyPoints)
        {
            MKeyPoint[] points = keyPoints.ToArray();
            foreach (var kp in points)
            {
                PointF point = new PointF(kp.Point.X, kp.Point.Y);
                if (!_slamPoints.Any(p => Math.Abs(p.X - point.X) < 1.0f &&
                                        Math.Abs(p.Y - point.Y) < 1.0f))
                {
                    _slamPoints.Add(point);
                }
            }
            FilterVerticalPoints();
            RepelPointsFromRobot();
        }

        private static void FilterVerticalPoints()
        {
            _slamPoints = _slamPoints
                .Where(p => Math.Abs(p.X - _currentRobotPosition.X) < 50)
                .ToList();
        }

        private static void RepelPointsFromRobot()
        {
            float repelRadius = 30f;
            var newPoints = new List<PointF>();

            foreach (var point in _slamPoints)
            {
                float dx = point.X - _currentRobotPosition.X;
                float dy = point.Y - _currentRobotPosition.Y;
                float distance = (float)Math.Sqrt(dx * dx + dy * dy);

                if (distance < repelRadius && distance > 0)
                {
                    float repelForce = (repelRadius - distance) / distance;
                    newPoints.Add(new PointF(
                        point.X + dx * repelForce,
                        point.Y + dy * repelForce));
                }
                else
                {
                    newPoints.Add(point);
                }
            }
            _slamPoints = newPoints;
        }

        private static void UpdateSlamMap()
        {
            if (_slamPoints.Count == 0) return;

            float minX = _slamPoints.Min(p => p.X);
            float maxX = _slamPoints.Max(p => p.X);
            float minY = _slamPoints.Min(p => p.Y);
            float maxY = _slamPoints.Max(p => p.Y);

            float centerX = (minX + maxX) / 2;
            float centerY = (minY + maxY) / 2;

            for (int i = 0; i < _slamPoints.Count; i++)
            {
                _slamPoints[i] = new PointF(
                    _slamPoints[i].X - centerX,
                    _slamPoints[i].Y - centerY);
            }

            _currentRobotPosition = new PointF(
                _currentRobotPosition.X - centerX,
                _currentRobotPosition.Y - centerY);

            for (int i = 0; i < _robotPath.Count; i++)
            {
                _robotPath[i] = new PointF(
                    _robotPath[i].X - centerX,
                    _robotPath[i].Y - centerY);
            }
        }

        private static byte[] GenerateSlamMapImage()
        {
            if (_slamPoints.Count == 0)
                return new byte[0];

            int width = 800;
            int height = 600;
            using (Mat map = new Mat(height, width, DepthType.Cv8U, 3))
            {
                map.SetTo(new MCvScalar(255, 255, 255));

                CvInvoke.Line(map,
                    new Point(width / 2, 0),
                    new Point(width / 2, height),
                    new MCvScalar(0, 0, 0), 1);
                CvInvoke.Line(map,
                    new Point(0, height / 2),
                    new Point(width, height / 2),
                    new MCvScalar(0, 0, 0), 1);

                foreach (var point in _slamPoints)
                {
                    int x = (int)(point.X * Scale) + width / 2;
                    int y = (int)(point.Y * Scale) + height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height)
                    {
                        CvInvoke.Circle(map,
                            new Point(x, y),
                            3,
                            new MCvScalar(0, 255, 0),
                            -1);
                    }
                }

                for (int i = 1; i < _robotPath.Count; i++)
                {
                    Point prev = new Point(
                        (int)(_robotPath[i - 1].X * Scale) + width / 2,
                        (int)(_robotPath[i - 1].Y * Scale) + height / 2);
                    Point current = new Point(
                        (int)(_robotPath[i].X * Scale) + width / 2,
                        (int)(_robotPath[i].Y * Scale) + height / 2);

                    if (prev.X >= 0 && prev.X < width && prev.Y >= 0 && prev.Y < height &&
                        current.X >= 0 && current.X < width && current.Y >= 0 && current.Y < height)
                    {
                        CvInvoke.Line(map, prev, current, new MCvScalar(255, 0, 0), 2);
                    }
                }

                Point robotPos = new Point(
                    (int)(_currentRobotPosition.X * Scale) + width / 2,
                    (int)(_currentRobotPosition.Y * Scale) + height / 2);

                if (robotPos.X >= 0 && robotPos.X < width &&
                    robotPos.Y >= 0 && robotPos.Y < height)
                {
                    CvInvoke.Circle(map, robotPos, 5, new MCvScalar(0, 0, 255), -1);
                    Point arrowEnd = new Point(
                        robotPos.X + (int)(20 * Math.Cos(_currentRobotAngleY)),
                        robotPos.Y + (int)(20 * Math.Sin(_currentRobotAngleY)));
                    CvInvoke.ArrowedLine(map, robotPos, arrowEnd, new MCvScalar(0, 0, 255), 2);
                }

                return map.ToImage<Bgr, byte>().ToJpegData();
            }
        }

        private static float GetNextX()
        {
            float moveDistance = 0.5f;
            float nextX = _currentRobotPosition.X + moveDistance * (float)Math.Cos(_currentRobotAngleY);

            if (_slamPoints.Any(p =>
                Math.Abs(p.X - nextX) < 10 &&
                Math.Abs(p.Y - _currentRobotPosition.Y) < 10))
            {
                return nextX;
            }
            return _currentRobotPosition.X;
        }

        private static float GetNextY()
        {
            float moveDistance = 0.5f;
            float nextY = _currentRobotPosition.Y + moveDistance * (float)Math.Sin(_currentRobotAngleY);

            if (_slamPoints.Any(p =>
                Math.Abs(p.X - _currentRobotPosition.X) < 10 &&
                Math.Abs(p.Y - nextY) < 10))
            {
                return nextY;
            }
            return _currentRobotPosition.Y;
        }
    }
}