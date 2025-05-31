using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Collections.Concurrent;
using System.Linq;
using System;
using System.Collections.Generic;

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

    public class Point3D
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }

    public static class SlamProcessor
    {
        private static ConcurrentBag<Point3D> _slamPoints = new ConcurrentBag<Point3D>();
        private static ConcurrentBag<PointF> _robotPath = new ConcurrentBag<PointF>();
        private static PointF _currentRobotPosition = new PointF(0, 0);
        private static float _currentRobotAngle = 0;
        private static readonly object _positionLock = new object();
        private static readonly object _slamPointsLock = new object();
        private static readonly object _robotPathLock = new object();
        private static readonly SIFT _siftDetector = new SIFT();
        private const float _maxDistanceFromPath = 5f;
        private const float _minDistanceBetweenPoints = 0.3f;
        private const float _pathSimplificationThreshold = 0.1f;
        private const double _baseline = 1.0;
        private const double _minDisparity = 1.0;

        public static StereoFrameResponse ProcessFrame(StereoFrameRequest request)
        {
            PointF currentPosition = new PointF((float)request.RobotX, (float)request.RobotY);
            float currentAngle = (float)request.RobotAngleY;

            lock (_positionLock)
            {
                _currentRobotPosition = currentPosition;
                _currentRobotAngle = currentAngle;
            }

            UpdateRobotPath(currentPosition);

            Mat leftImage = new Mat();
            CvInvoke.Imdecode(request.LeftImageData, ImreadModes.Color, leftImage);
            Mat rightImage = new Mat();
            CvInvoke.Imdecode(request.RightImageData, ImreadModes.Color, rightImage);

            // Получаем ключевые точки для левого изображения
            VectorOfKeyPoint leftKeyPoints = DetectKeypoints(leftImage);
            Mat processedLeft = DrawKeypoints(leftImage, leftKeyPoints);
            Mat processedRight = ProcessRightImage(rightImage);

            Mat disparity = ComputeDisparity(leftImage, rightImage);
            UpdateSlamPointsWithStereo(disparity, leftKeyPoints, currentPosition,
                _currentRobotAngle, request.LeftCameraInfo);

            return new StereoFrameResponse
            {
                ProcessedLeftImage = processedLeft.ToImage<Bgr, byte>().ToJpegData(),
                ProcessedRightImage = processedRight.ToImage<Bgr, byte>().ToJpegData(),
                SlamMapImage = GenerateSlamMapImage()
            };
        }

        private static VectorOfKeyPoint DetectKeypoints(Mat image)
        {
            if (image.IsEmpty)
                return new VectorOfKeyPoint();

            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayImage, grayImage);

            VectorOfKeyPoint keyPoints = new VectorOfKeyPoint();
            using (Mat descriptors = new Mat())
            {
                _siftDetector.DetectAndCompute(grayImage, null, keyPoints, descriptors, false);
                return keyPoints;
            }
        }

        private static Mat DrawKeypoints(Mat image, VectorOfKeyPoint keyPoints)
        {
            if (image.IsEmpty)
                return new Mat();

            Mat result = image.Clone();
            if (keyPoints.Size > 0)
            {
                Features2DToolbox.DrawKeypoints(
                    image,
                    keyPoints,
                    result,
                    new Bgr(Color.Red),
                    Features2DToolbox.KeypointDrawType.Default);
            }
            return result;
        }

        private static Mat ComputeDisparity(Mat leftImage, Mat rightImage)
        {
            Mat leftGray = new Mat();
            Mat rightGray = new Mat();
            CvInvoke.CvtColor(leftImage, leftGray, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(rightImage, rightGray, ColorConversion.Bgr2Gray);

            using StereoSGBM sgbm = new StereoSGBM(
                minDisparity: 0,
                numDisparities: 64,
                blockSize: 11,
                disp12MaxDiff: -1,
                preFilterCap: 63,
                uniquenessRatio: 10,
                speckleWindowSize: 100,
                speckleRange: 32,
                mode: StereoSGBM.Mode.SGBM);

            Mat disparity = new Mat();
            sgbm.Compute(leftGray, rightGray, disparity);
            return disparity;
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

        private static Mat ProcessLeftImage(Mat image, PointF currentPosition, CameraInfo cameraInfo)
        {
            if (image.IsEmpty || cameraInfo == null)
                return new Mat();

            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayImage, grayImage);

            VectorOfKeyPoint keyPoints = new VectorOfKeyPoint();
            using (Mat descriptors = new Mat())
            {
                _siftDetector.DetectAndCompute(grayImage, null, keyPoints, descriptors, false);
                Mat result = image.Clone();
                Features2DToolbox.DrawKeypoints(
                    image,
                    keyPoints,
                    result,
                    new Bgr(Color.Red),
                    Features2DToolbox.KeypointDrawType.Default);
                return result;
            }
        }

        private static Mat ProcessRightImage(Mat image)
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
                Features2DToolbox.DrawKeypoints(
                    image,
                    keyPoints,
                    result,
                    new Bgr(Color.Red),
                    Features2DToolbox.KeypointDrawType.Default);
                return result;
            }
        }

        private static void UpdateSlamPointsWithStereo(Mat disparity, VectorOfKeyPoint leftKeyPoints,
            PointF currentPosition, float currentAngle, CameraInfo cameraInfo)
        {
            if (disparity.IsEmpty || cameraInfo == null || leftKeyPoints == null)
                return;

            double f = cameraInfo.FocalLength;
            double cx = cameraInfo.PrincipalPointX;
            double cy = cameraInfo.PrincipalPointY;
            double H = cameraInfo.Height;
            double OffsetX = cameraInfo.OffsetX;
            double OffsetZ = cameraInfo.OffsetZ;

            var newPoints = new ConcurrentBag<Point3D>();

            // Конвертируем disparity в 32-битный float
            using Mat disparity32f = new Mat();
            disparity.ConvertTo(disparity32f, DepthType.Cv32F, 1.0 / 16.0);

            // Получаем указатель на данные диспарантности
            var dataPointer = disparity32f.DataPointer;
            int step = disparity32f.Step / 4;  // Шаг в байтах делим на sizeof(float)

            unsafe
            {
                float* disparityPtr = (float*)dataPointer.ToPointer();

                for (int i = 0; i < leftKeyPoints.Size; i++)
                {
                    var kp = leftKeyPoints[i];
                    int u = (int)Math.Round(kp.Point.X);
                    int v = (int)Math.Round(kp.Point.Y);

                    if (u < 0 || u >= disparity32f.Cols || v < 0 || v >= disparity32f.Rows)
                        continue;

                    // Получаем значение диспарантности через указатель
                    float d = disparityPtr[v * step + u];

                    if (d < _minDisparity) continue;

                    double depth = (f * _baseline) / d;
                    double x_norm = (u - cx) / f;
                    double y_norm = (v - cy) / f;

                    double X_cam = x_norm * depth;
                    double Y_cam = y_norm * depth;
                    double Z_cam = depth;

                    double X_robot = Z_cam + OffsetZ;
                    double Y_robot = X_cam + OffsetX;
                    double Z_robot = -Y_cam + H;

                    Point3D point = new Point3D
                    {
                        X = (float)(currentPosition.X + X_robot * Math.Cos(currentAngle) - (float)(Y_robot * Math.Sin(currentAngle))),
                        Y = (float)(currentPosition.Y + X_robot * Math.Sin(currentAngle) + (float)(Y_robot * Math.Cos(currentAngle))),
                        Z = (float)Z_robot
                    };

                    PointF groundPoint = new PointF(point.X, point.Y);
                    if (IsPointValid(groundPoint, currentPosition))
                    {
                        newPoints.Add(point);
                    }
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
                return minDistance <= _maxDistanceFromPath && Distance(point, currentPosition) > _minDistanceBetweenPoints;
            }
        }

        private static void RemoveDistantPoints()
        {
            lock (_slamPointsLock)
                lock (_robotPathLock)
                {
                    var validPoints = new ConcurrentBag<Point3D>();
                    var pathList = _robotPath.ToList();

                    foreach (var slamPoint in _slamPoints)
                    {
                        PointF point = new PointF(slamPoint.X, slamPoint.Y);
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
                            validPoints.Add(slamPoint);
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
            List<Point3D> slamPointsCopy;
            List<PointF> robotPathCopy;
            PointF robotPositionCopy;
            float robotAngleCopy;

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
                robotAngleCopy = _currentRobotAngle;
            }

            if (slamPointsCopy.Count == 0 && robotPathCopy.Count == 0)
                return new byte[0];

            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;

            foreach (var point in slamPointsCopy.Select(p => new PointF(p.X, p.Y)))
            {
                minX = Math.Min(minX, point.X);
                maxX = Math.Max(maxX, point.X);
                minY = Math.Min(minY, point.Y);
                maxY = Math.Max(maxY, point.Y);
            }

            foreach (var point in robotPathCopy.Concat(new[] { robotPositionCopy }))
            {
                minX = Math.Min(minX, point.X);
                maxX = Math.Max(maxX, point.X);
                minY = Math.Min(minY, point.Y);
                maxY = Math.Max(maxY, point.Y);
            }

            float widthRange = maxX - minX;
            float heightRange = maxY - minY;
            float margin = 1f;
            float scale = Math.Min(780f / (widthRange + margin * 2), 580f / (heightRange + margin * 2));

            int imageWidth = 800;
            int imageHeight = 600;
            using (Mat map = new Mat(imageHeight, imageWidth, DepthType.Cv8U, 3))
            {
                map.SetTo(new MCvScalar(255, 255, 255));

                foreach (var slamPoint in slamPointsCopy)
                {
                    PointF point = new PointF(slamPoint.X, slamPoint.Y);
                    int x = (int)((point.X - minX + margin) * scale) + 10;
                    int y = (int)((point.Y - minY + margin) * scale) + 10;
                    if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight)
                    {
                        CvInvoke.Circle(map, new Point(x, y), 2, new MCvScalar(0, 255, 0), -1);
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

                    CvInvoke.Polylines(map, new VectorOfPoint(pathPoints), false, new MCvScalar(255, 0, 0), 2);
                }

                Point robotPos = new Point(
                    (int)((robotPositionCopy.X - minX + margin) * scale) + 10,
                    (int)((robotPositionCopy.Y - minY + margin) * scale) + 10);

                if (robotPos.X >= 0 && robotPos.X < imageWidth && robotPos.Y >= 0 && robotPos.Y < imageHeight)
                {
                    CvInvoke.Circle(map, robotPos, 5, new MCvScalar(0, 0, 255), -1);
                    Point arrowEnd = new Point(
                        robotPos.X + (int)(20 * Math.Cos(robotAngleCopy)),
                        robotPos.Y + (int)(20 * Math.Sin(robotAngleCopy)));
                    CvInvoke.ArrowedLine(map, robotPos, arrowEnd, new MCvScalar(0, 0, 255), 2);
                }

                return map.ToImage<Bgr, byte>().ToJpegData();
            }
        }
    }
}