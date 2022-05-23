package machine.controller;

import machine.components.DetectModel;
import machine.components.OcrModel;
import machine.components.ParkingStatusModel;
import machine.entity.ParkingStateResult;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.Resource;
import java.util.HashMap;

import static machine.helper.TensorflowHelper.formToImage;

@RestController
@RequestMapping(value = "/algorithm_server")
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
            var detect = detectModel.carPlateDetect(formToImage(file, 625, 625));

            var plate = ocrModel.carPlateRecognize(detect.plateImage);

            var m = new HashMap<String, Object>();
            m.put("left", Math.min(detect.leftTop.x, detect.leftBottom.x));
            m.put("right", Math.max(detect.rightTop.x, detect.rightBottom.x));
            m.put("top", Math.min(detect.leftTop.y, detect.rightTop.y));
            m.put("bottom", Math.max(detect.leftBottom.y, detect.rightBottom.y));
            m.put("plate", plate);
            m.put("confidence", 0);
            m.put("timeSpent", 0);
            return m;
        }).get();
    }

    @PostMapping(value = "/parking_status_recognize")
    public Object parkingStatusRecognize(@RequestParam("file") MultipartFile file) throws Exception {
        return executor.submit(() -> {
            var begin = System.currentTimeMillis();
            var status = parkingStatusModel.predict(formToImage(file, 96, 96));
            return new ParkingStateResult(status, System.currentTimeMillis() - begin);
        }).get();
    }

    @PostMapping(value = "/hpc")
    public Object hpc(@RequestParam("file") MultipartFile file) {
        return "";
    }
}
