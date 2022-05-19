package machine.controller;

import machine.helper.TensorflowHelper;
import machine.models.DetectModel;
import machine.models.OcrModel;
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

    @PostMapping("/car_plate")
    public Object detect(@RequestParam("file") MultipartFile file) throws IOException {
        var image = TensorflowHelper.formToImage(file);

        var plateImage = detectModel.carPlateDetect(image);

        return ocrModel.carPlateRecognize(plateImage);
    }
}
