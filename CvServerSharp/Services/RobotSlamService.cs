using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Collections.Concurrent;
using System.Linq;

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
    }

    public class CameraInfo
    {
        public double FocalLength { get; set; }
        public double PrincipalPointX { get; set; }
        public double PrincipalPointY { get; set; }
        public double OffsetX { get; set; }
        public double OffsetZ { get; set; }
        public double Height { get; set; }
    }

    public static class SlamProcessor
    {
        private static ConcurrentBag<PointF> _slamPoints = new ConcurrentBag<PointF>();
        private static ConcurrentBag<PointF> _robotPath = new ConcurrentBag<PointF>();
        private static PointF _currentRobotPosition = new PointF(0, 0);
        private static float _currentRobotAngleY = 0;
        private static readonly object _positionLock = new object();
        private static readonly object _slamPointsLock = new object();
        private static readonly object _robotPathLock = new object();
        private static readonly SIFT _siftDetector = new SIFT();
        private const float _maxDistanceFromPath = 5f;
        private const float _minDistanceBetweenPoints = 0.3f;
        private const float _pathSimplificationThreshold = 0.1f;

        public static StereoFrameResponse ProcessFrame(StereoFrameRequest request)
        {
            PointF currentPosition = new PointF((float)request.RobotX, (float)request.RobotY);
            float currentAngleY = (float)request.RobotAngleY;

            lock (_positionLock)
            {
                _currentRobotPosition = currentPosition;
                _currentRobotAngleY = currentAngleY;
            }

            UpdateRobotPath(currentPosition);

            Mat leftImage = new Mat();
            CvInvoke.Imdecode(request.LeftImageData, ImreadModes.Color, leftImage);
            Mat rightImage = new Mat();
            CvInvoke.Imdecode(request.RightImageData, ImreadModes.Color, rightImage);

            Mat processedLeft = ProcessImage(leftImage, currentPosition);
            Mat processedRight = ProcessImage(rightImage, currentPosition);

            return new StereoFrameResponse
            {
                ProcessedLeftImage = processedLeft.ToImage<Bgr, byte>().ToJpegData(),
                ProcessedRightImage = processedRight.ToImage<Bgr, byte>().ToJpegData(),
                SlamMapImage = GenerateSlamMapImage()
            };
        }

        private static void UpdateRobotPath(PointF currentPosition)
        {
            lock (_robotPathLock)
            {
                if (!_robotPath.Any() ||
                    Distance(_robotPath.Last(), currentPosition) > _pathSimplificationThreshold)
                {
                    _robotPath.Add(currentPosition);
                }
            }
        }

        private static Mat ProcessImage(Mat image, PointF currentPosition)
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

                    UpdateSlamPoints(keyPoints, currentPosition);
                }
                return result;
            }
        }

        private static void UpdateSlamPoints(VectorOfKeyPoint keyPoints, PointF currentPosition)
        {
            MKeyPoint[] points = keyPoints.ToArray();
            var newPoints = new ConcurrentBag<PointF>();

            foreach (var kp in points)
            {
                PointF point = new PointF(currentPosition.X + kp.Point.X / 100f, currentPosition.Y + kp.Point.Y / 100f);

                if (IsPointValid(point, currentPosition))
                {
                    newPoints.Add(point);
                }
            }

            if (newPoints.Count > 0)
            {
                lock (_slamPointsLock)
                {
                    foreach (var point in newPoints)
                    {
                        _slamPoints.Add(point);
                    }
                    RemoveDistantPoints();
                }
            }
        }

        private static bool IsPointValid(PointF point, PointF currentPosition)
        {
            lock (_robotPathLock)
            {
                if (!_robotPath.Any())
                    return false;

                float minDistance = _robotPath.Min(p => Distance(p, point));
                return minDistance <= _maxDistanceFromPath;
            }
        }

        private static void RemoveDistantPoints()
        {
            lock (_slamPointsLock)
                lock (_robotPathLock)
                {
                    var validPoints = new ConcurrentBag<PointF>();
                    var pathList = _robotPath.ToList();

                    foreach (var point in _slamPoints)
                    {
                        bool keepPoint = false;
                        foreach (var pathPoint in pathList)
                        {
                            if (Distance(point, pathPoint) <= _maxDistanceFromPath)
                            {
                                keepPoint = true;
                                break;
                            }
                        }

                        if (keepPoint)
                        {
                            validPoints.Add(point);
                        }
                    }

                    _slamPoints = validPoints;
                }
        }

        private static float Distance(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        private static byte[] GenerateSlamMapImage()
        {
            List<PointF> slamPointsCopy;
            List<PointF> robotPathCopy;
            PointF robotPositionCopy;
            float robotAngleYCopy;

            lock (_slamPointsLock)
            {
                slamPointsCopy = _slamPoints.ToList();
            }

            lock (_robotPathLock)
            {
                robotPathCopy = _robotPath.ToList();
            }

            lock (_positionLock)
            {
                robotPositionCopy = _currentRobotPosition;
                robotAngleYCopy = _currentRobotAngleY;
            }

            if (slamPointsCopy.Count == 0 && robotPathCopy.Count == 0)
                return new byte[0];

            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;

            foreach (var point in slamPointsCopy.Concat(robotPathCopy))
            {
                minX = Math.Min(minX, point.X);
                maxX = Math.Max(maxX, point.X);
                minY = Math.Min(minY, point.Y);
                maxY = Math.Max(maxY, point.Y);
            }

            minX = Math.Min(minX, robotPositionCopy.X);
            maxX = Math.Max(maxX, robotPositionCopy.X);
            minY = Math.Min(minY, robotPositionCopy.Y);
            maxY = Math.Max(maxY, robotPositionCopy.Y);

            float widthRange = maxX - minX;
            float heightRange = maxY - minY;
            float margin = 1f;
            float scale = Math.Min(780f / (widthRange + margin * 2), 580f / (heightRange + margin * 2));

            int imageWidth = 800;
            int imageHeight = 600;
            using (Mat map = new Mat(imageHeight, imageWidth, DepthType.Cv8U, 3))
            {
                map.SetTo(new MCvScalar(255, 255, 255));

                foreach (var point in slamPointsCopy)
                {
                    int x = (int)((point.X - minX + margin) * scale) + 10;
                    int y = (int)((point.Y - minY + margin) * scale) + 10;
                    if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight)
                    {
                        CvInvoke.Circle(map, new Point(x, y), 3, new MCvScalar(0, 255, 0), -1);
                    }
                }

                if (robotPathCopy.Count > 1)
                {
                    Point[] pathPoints = robotPathCopy
                        .Select(p => new Point(
                            (int)((p.X - minX + margin) * scale) + 10,
                            (int)((p.Y - minY + margin) * scale) + 10))
                        .Where(p => p.X >= 0 && p.X < imageWidth && p.Y >= 0 && p.Y < imageHeight)
                        .ToArray();

                    if (pathPoints.Length > 1)
                    {
                        CvInvoke.Polylines(map, new VectorOfPoint(pathPoints), false, new MCvScalar(255, 0, 0), 2);
                    }
                }

                Point robotPos = new Point(
                    (int)((robotPositionCopy.X - minX + margin) * scale) + 10,
                    (int)((robotPositionCopy.Y - minY + margin) * scale) + 10);

                if (robotPos.X >= 0 && robotPos.X < imageWidth && robotPos.Y >= 0 && robotPos.Y < imageHeight)
                {
                    CvInvoke.Circle(map, robotPos, 5, new MCvScalar(0, 0, 255), -1);
                    Point arrowEnd = new Point(
                        robotPos.X + (int)(20 * Math.Cos(robotAngleYCopy)),
                        robotPos.Y + (int)(20 * Math.Sin(robotAngleYCopy)));
                    CvInvoke.ArrowedLine(map, robotPos, arrowEnd, new MCvScalar(0, 0, 255), 2);
                }

                return map.ToImage<Bgr, byte>().ToJpegData();
            }
        }
    }
}