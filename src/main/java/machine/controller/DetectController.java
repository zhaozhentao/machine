package machine.controller;

import machine.helper.TensorflowHelper;
import machine.models.DetectModel;
import machine.models.OcrModel;
import machine.models.ParkingStatusModel;
import machine.pojo.ParkingStateResult;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.Resource;
import java.io.IOException;

@RestController
public class DetectController {

    @Resource
    private DetectModel detectModel;

    @Resource
    private OcrModel ocrModel;

    @Resource
    private ParkingStatusModel parkingStatusModel;

    @PostMapping("/lpr")
    public Object detect(@RequestParam("file") MultipartFile file) throws IOException {
        var image = TensorflowHelper.formToImage(file, 625, 625);

        var plateImage = detectModel.carPlateDetect(image);

        return ocrModel.carPlateRecognize(plateImage);
    }

    @PostMapping(value = "/parking_status_recognize")
    public Object parkingStatusRecognize(@RequestParam("file") MultipartFile file) throws IOException {
        var image = TensorflowHelper.formToImage(file, 96, 96);

        return new ParkingStateResult(parkingStatusModel.predict(image));
    }
}
