package machine.controller;

import machine.components.DetectModel;
import machine.components.OcrModel;
import machine.components.ParkingStatusModel;
import machine.pojo.ParkingStateResult;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.Resource;

import static machine.helper.TensorflowHelper.formToImage;

@RestController
public class DetectController {

    @Resource
    OcrModel ocrModel;

    @Resource
    DetectModel detectModel;

    @Resource
    ParkingStatusModel parkingStatusModel;

    @Resource
    ThreadPoolTaskExecutor executor;

    @PostMapping("/lpr")
    public Object detect(@RequestParam("file") MultipartFile file) throws Exception {
        return executor.submit(() -> {
            var plateImage = detectModel.carPlateDetect(formToImage(file, 625, 625));

            return ocrModel.carPlateRecognize(plateImage);
        }).get();
    }

    @PostMapping(value = "/parking_status_recognize")
    public Object parkingStatusRecognize(@RequestParam("file") MultipartFile file) throws Exception {
        return executor.submit(() -> new ParkingStateResult(
                parkingStatusModel.predict(formToImage(file, 96, 96))
            )
        ).get();
    }

    @PostMapping(value = "/hpc")
    public Object hpc(@RequestParam("file") MultipartFile file) {
        return "";
    }
}
