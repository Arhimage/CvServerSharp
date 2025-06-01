using Microsoft.AspNetCore.Mvc;
using System.IO;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Serv;

namespace RobotSlamServer
{
    [ApiController]
    [Route("api/slam")]
    public class RobotSlamController : ControllerBase
    {
        //private readonly SlamProcessor _slamService;
        private readonly ILogger<RobotSlamController> _logger;

        public RobotSlamController(ILogger<RobotSlamController> logger)
        {
            //_slamService = slamService;
            _logger = logger;
        }

        [HttpPost("processstereoframe")]
        public async Task<IActionResult> ProcessStereoFrame(
            [FromForm] float robotX,
            [FromForm] float robotY,
            [FromForm] float robotAngleX,  // Добавлен угол X для 3D-ориентации
            [FromForm] float robotAngleY,  // Добавлен угол Y для 3D-ориентации
            [FromForm] float robotAngleZ,  // Переименовано для соответствия 3D-ориентации
            [FromForm] IFormFile leftImageData,
            [FromForm] IFormFile rightImageData,
            [FromForm] double leftCameraOffsetX,
            [FromForm] double leftCameraOffsetY,
            [FromForm] double leftCameraOffsetZ,
            [FromForm] double leftCameraFocalLength,
            [FromForm] double leftCameraPrincipalPointX,
            [FromForm] double leftCameraPrincipalPointY,
            [FromForm] double rightCameraOffsetX,
            [FromForm] double rightCameraOffsetY,
            [FromForm] double rightCameraOffsetZ,
            [FromForm] double rightCameraFocalLength,
            [FromForm] double rightCameraPrincipalPointX,
            [FromForm] double rightCameraPrincipalPointY)
        {
            // Проверка наличия изображений
            if (leftImageData == null || leftImageData.Length == 0 ||
                rightImageData == null || rightImageData.Length == 0)
            {
                return BadRequest("Оба изображения (левое и правое) обязательны для обработки");
            }

            try
            {
                // Чтение изображений в массивы байтов
                byte[] leftImageBytes, rightImageBytes;

                using (var leftStream = new MemoryStream())
                {
                    await leftImageData.CopyToAsync(leftStream);
                    leftImageBytes = leftStream.ToArray();
                }
                using (var rightStream = new MemoryStream())
                {
                    await rightImageData.CopyToAsync(rightStream);
                    rightImageBytes = rightStream.ToArray();
                }

                // Формирование обновленных объектов информации о камерах
                var leftCameraInfo = new CameraInfo
                {
                    OffsetX = leftCameraOffsetX,
                    Height = leftCameraOffsetY,
                    OffsetZ = leftCameraOffsetZ,
                    FocalLength = leftCameraFocalLength,
                    PrincipalPointX = leftCameraPrincipalPointX,
                    PrincipalPointY = leftCameraPrincipalPointY
                };

                var rightCameraInfo = new CameraInfo
                {
                    OffsetX = rightCameraOffsetX,
                    Height = rightCameraOffsetY,
                    OffsetZ = rightCameraOffsetZ,
                    FocalLength = rightCameraFocalLength,
                    PrincipalPointX = rightCameraPrincipalPointX,
                    PrincipalPointY = rightCameraPrincipalPointY
                };

                // Создание запроса для SLAM обработки
                StereoFrameRequest stereoRequest = new StereoFrameRequest
                {
                    RobotX = robotX,
                    RobotY = robotY,
                    RobotAngleZ = robotAngleY,
                    LeftCameraInfo = leftCameraInfo,
                    LeftImageData = leftImageBytes,
                    RightCameraInfo = rightCameraInfo,
                    RightImageData = rightImageBytes
                };

                // Передача запроса в сервис для обработки
                var result = SlamProcessor.ProcessFrame(stereoRequest);

                // Возвращаем ответ с обработанными данными и изображениями
                return Ok(result);
            }
            catch (System.Exception ex)
            {
                _logger.LogError(ex, "Ошибка обработки стерео кадров: {Message}", ex.Message);
                return StatusCode(500, $"Внутренняя ошибка сервера при обработке стереокадров: {ex.Message}");
            }
        }

        [HttpPost("qr")]
        public async Task<IActionResult> ProcessQR(
            [FromForm] IFormFile leftImageData,
            [FromForm] IFormFile rightImageData)
        {
            if (leftImageData == null || leftImageData.Length == 0 ||
                rightImageData == null || rightImageData.Length == 0)
            {
                return BadRequest("Оба изображения (левое и правое) обязательны для обработки");
            }

            try
            {
                byte[] leftImageBytes, rightImageBytes;

                using (var leftStream = new MemoryStream())
                {
                    await leftImageData.CopyToAsync(leftStream);
                    leftImageBytes = leftStream.ToArray();
                }
                using (var rightStream = new MemoryStream())
                {
                    await rightImageData.CopyToAsync(rightStream);
                    rightImageBytes = rightStream.ToArray();
                }

                Serv.StereoFrameRequest stereoRequest = new Serv.StereoFrameRequest
                {
                    LeftImageData = leftImageBytes,
                    RightImageData = rightImageBytes
                };

                var result = new QRService().ProcessStereoFrame(stereoRequest);

                return Ok(result);
            }
            catch (System.Exception ex)
            {
                _logger.LogError(ex, "Ошибка обработки стерео кадров: {Message}", ex.Message);
                return StatusCode(500, $"Внутренняя ошибка сервера при обработке стереокадров: {ex.Message}");
            }
        }

    }
}