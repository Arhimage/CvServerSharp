using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Drawing;

namespace Serv;

public class StereoFrameRequest
{
    public byte[] LeftImageData { get; set; }
    public byte[] RightImageData { get; set; }
}

public class QRService
{
    public StereoFrameRequest ProcessStereoFrame(StereoFrameRequest request)
    {
        return new StereoFrameRequest
        {
            LeftImageData = ProcessImage(request.LeftImageData),
            RightImageData = ProcessImage(request.RightImageData)
        };
    }

    private byte[] ProcessImage(byte[] imageData)
    {
        if (imageData == null || imageData.Length == 0)
            return imageData;

        try
        {
            using (Mat img = new Mat())
            using (QRCodeDetector qrDetector = new QRCodeDetector())
            {
                CvInvoke.Imdecode(imageData, ImreadModes.Color, img);

                if (img.IsEmpty)
                    return imageData;

                VectorOfPoint points = new VectorOfPoint();
                bool found = qrDetector.Detect(img, points);

                if (found && points.Length > 0)
                {
                    Point[] qrPoints = points.ToArray();
                    string decodedText = qrDetector.Decode(img, points);

                    // Рисуем рамку вокруг QR-кода
                    for (int i = 0; i < qrPoints.Length; i++)
                    {
                        CvInvoke.Line(
                            img,
                            qrPoints[i],
                            qrPoints[(i + 1) % qrPoints.Length],
                            new MCvScalar(0, 255, 0),
                            3
                        );
                    }

                    // Добавляем текст
                    if (!string.IsNullOrEmpty(decodedText))
                    {
                        Point textOrigin = new Point(
                            qrPoints[0].X,
                            Math.Max(20, qrPoints[0].Y - 10)
                        );

                        CvInvoke.PutText(
                            img,
                            decodedText,
                            textOrigin,
                            FontFace.HersheySimplex,
                            0.8,
                            new MCvScalar(0, 0, 255),
                            2
                        );
                    }
                }

                using (VectorOfByte vb = new VectorOfByte())
                {
                    CvInvoke.Imencode(".png", img, vb);
                    return vb.ToArray();
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error processing image: {ex.Message}");
            return imageData;
        }
    }
}