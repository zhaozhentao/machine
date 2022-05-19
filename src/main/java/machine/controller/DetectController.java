package machine.controller;

import machine.components.DetectModel;
import machine.components.OcrModel;
import machine.components.ParkingStatusModel;
import machine.helper.TensorflowHelper;
import machine.pojo.ParkingStateResult;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.Resource;

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
            var image = TensorflowHelper.formToImage(file, 625, 625);

            var plateImage = detectModel.carPlateDetect(image);

            return ocrModel.carPlateRecognize(plateImage);
        }).get();
    }

    @PostMapping(value = "/parking_status_recognize")
    public Object parkingStatusRecognize(@RequestParam("file") MultipartFile file) throws Exception {
        return executor.submit(() -> {
            var image = TensorflowHelper.formToImage(file, 96, 96);

            return new ParkingStateResult(parkingStatusModel.predict(image));
        }).get();
    }

    @PostMapping(value = "/hpc")
    public Object hpc(@RequestParam("file") MultipartFile file) {
        return "";
    }
}
