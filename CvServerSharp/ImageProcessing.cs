//using Emgu.CV;
//using Emgu.CV.CvEnum;
//using Emgu.CV.Features2D;
//using Emgu.CV.Structure;
//using Emgu.CV.Util;
//using System.Drawing;

//namespace RobotSlamServer
//{
//    public static class ImageProcessing
//    {
//        public static void ProcessImage(Mat image, out VectorOfKeyPoint keypoints, out Mat descriptors)
//        {
//            Console.WriteLine("[ProcessImage] Начало обработки изображения");
//            // Проверка входного изображения
//            if (image == null || image.IsEmpty)
//            {
//                throw new ArgumentException("Входное изображение пустое или null");
//            }

//            using (var orb = new ORB())
//            {
//                Mat gray = new Mat();
//                CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);
//                keypoints = new VectorOfKeyPoint();
//                descriptors = new Mat();
//                orb.DetectAndCompute(gray, null, keypoints, descriptors, false);
//                Console.WriteLine($"[ProcessImage] Найдено {keypoints.Size} ключевых точек");
//            }
//        }

//        public static VectorOfDMatch MatchDescriptors(Mat des1, Mat des2)
//        {
//            Console.WriteLine("[MatchDescriptors] Начинаем сопоставление дескрипторов");
//            // Проверки на null и пустоту
//            if (des1 == null || des1.IsEmpty || des2 == null || des2.IsEmpty)
//            {
//                throw new ArgumentException("Один или оба дескриптора пусты или null");
//            }

//            using (var matcher = new BFMatcher(DistanceType.Hamming))
//            {
//                VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
//                matcher.KnnMatch(des1, des2, matches, 2);

//                // Применение теста отношения Лоу для фильтрации совпадений
//                List<MDMatch> goodMatches = new List<MDMatch>();
//                MDMatch[][] matchesArray = matches.ToArrayOfArray();

//                for (int i = 0; i < matchesArray.Length; i++)
//                {
//                    if (matchesArray[i].Length >= 2)
//                    {
//                        MDMatch m = matchesArray[i][0];
//                        MDMatch n = matchesArray[i][1];

//                        if (m.Distance < 0.75 * n.Distance)
//                        {
//                            goodMatches.Add(m);
//                        }
//                    }
//                }

//                VectorOfDMatch result = new VectorOfDMatch();
//                result.Push(goodMatches.ToArray());

//                Console.WriteLine($"[MatchDescriptors] Найдено {result.Size} качественных совпадений");
//                return result;
//            }
//        }

//        public static Mat EstimateTransform(VectorOfKeyPoint kp1, VectorOfKeyPoint kp2, VectorOfDMatch matches)
//        {
//            Console.WriteLine("[EstimateTransform] Оценка аффинного преобразования");
//            if (matches.Size < 4)
//            {
//                Console.WriteLine("[EstimateTransform] Недостаточно совпадений, невозможно оценить преобразование");
//                return null;
//            }

//            List<PointF> pts1 = new List<PointF>();
//            List<PointF> pts2 = new List<PointF>();

//            MDMatch[] matchArray = matches.ToArray();
//            foreach (var match in matchArray)
//            {
//                pts1.Add(kp1[match.QueryIdx].Point);
//                pts2.Add(kp2[match.TrainIdx].Point);
//            }

//            if (pts1.Count < 4 || pts2.Count < 4)
//            {
//                Console.WriteLine("[EstimateTransform] Недостаточно корректных точек для оценки преобразования");
//                return null;
//            }

//            // Преобразуем списки PointF в массивы точек для Emgu CV
//            PointF[] srcPoints = pts1.ToArray();
//            PointF[] dstPoints = pts2.ToArray();

//            // Исправленный вызов EstimateAffine2D с параметром для маски
//            Mat inlierMask = new Mat();
//            Mat M = CvInvoke.EstimateAffine2D(srcPoints, dstPoints, inlierMask, RobustEstimationAlgorithm.Ransac, 3.0, 2000, 0.99, 10);

//            if (M == null || M.IsEmpty)
//            {
//                Console.WriteLine("[EstimateTransform] Оценить преобразование не удалось");
//                return null;
//            }

//            Console.WriteLine("[EstimateTransform] Преобразование оценено успешно");
//            return M;
//        }

//        public static ((double dx, double dy), double angle) ComputeOffsetAndRotation(Mat M)
//        {
//            Console.WriteLine("[ComputeOffsetAndRotation] Вычисление смещения и угла поворота");
//            if (M == null || M.IsEmpty)
//            {
//                Console.WriteLine("[ComputeOffsetAndRotation] Матрица преобразования пуста. Возвращаем нулевые значения");
//                return ((0.0, 0.0), 0.0);
//            }

//            // Исправленное получение значений из матрицы
//            double dx;
//            double dy;
//            double m00;
//            double m10;

//            using (Matrix<double> mat = new Matrix<double>(M.Rows, M.Cols, M.NumberOfChannels))
//            {
//                M.CopyTo(mat);
//                dx = mat[0, 2];
//                dy = mat[1, 2];
//                m00 = mat[0, 0];
//                m10 = mat[1, 0];
//            }

//            double angleRad = Math.Atan2(m10, m00);
//            double angleDeg = angleRad * (180.0 / Math.PI);

//            Console.WriteLine($"[ComputeOffsetAndRotation] Смещение: dx={dx}, dy={dy}, угол={angleDeg}°");
//            return ((dx, dy), angleDeg);
//        }

//        public static Mat CreateSlamImage(Mat image, VectorOfKeyPoint keypoints = null)
//        {
//            Console.WriteLine("[CreateSlamImage] Создание SLAM изображения");
//            if (image == null || image.IsEmpty)
//            {
//                throw new ArgumentException("Входное изображение пусто или null");
//            }

//            Mat slamImage = new Mat();
//            image.CopyTo(slamImage);

//            // Если есть ключевые точки, нарисуем их на slam изображении
//            if (keypoints != null && keypoints.Size > 0)
//            {
//                Features2DToolbox.DrawKeypoints(slamImage, keypoints, slamImage,
//                    new Bgr(0, 255, 0), Features2DToolbox.KeypointDrawType.DrawRichKeypoints);
//            }

//            return slamImage;
//        }
//    }
//}