package machine.controller;

import machine.components.DetectModel;
import machine.components.OcrModel;
import machine.components.ParkingStatusModel;
import machine.entity.ParkingStateResult;
import machine.helper.TensorflowHelper;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.*;
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

    private HashMap<String, Object> result(Object data) {
        return new HashMap<>(4) {{
            put("resultCode", 0);
            put("msg", "Success");
            put("msgParameters", "Success");
            put("data", data);
        }};
    }

    @PostMapping("/lpr")
    public Object detect(@RequestParam("file") MultipartFile file) throws Exception {
        return executor.submit(() -> {
            var begin = System.currentTimeMillis();

            var resultMap = formToImage(file, 320, 320);
            var detect = detectModel.carPlateDetect(resultMap.resized, resultMap.raw);

            var plate = ocrModel.carPlateRecognize(detect.plateImage);

            return result(new HashMap<String, Object>(8) {{
                put("left", Math.min(detect.leftTop.x, detect.leftBottom.x));
                put("right", Math.max(detect.rightTop.x, detect.rightBottom.x));
                put("top", Math.min(detect.leftTop.y, detect.rightTop.y));
                put("bottom", Math.max(detect.leftBottom.y, detect.rightBottom.y));
                put("plate", plate);
                put("confidence", 0);
                put("timeSpent", System.currentTimeMillis() - begin);
            }});
        }).get();
    }

    @PostMapping(value = "/parking_status_recognize")
    public Object parkingStatusRecognize(@RequestParam("file") MultipartFile file) throws Exception {
        return executor.submit(() -> {
            var begin = System.currentTimeMillis();
            var result = formToImage(file, 96, 96);
            result.raw.release();
            var status = parkingStatusModel.predict(result.resized);
            return result(new ParkingStateResult(status, System.currentTimeMillis() - begin));
        }).get();
    }

    @PostMapping(value = "/hpc")
    public Object hpc(@RequestParam("file") MultipartFile file) {
        return "";
    }

    @GetMapping("license_request_code")
    public Object code() {
        return result(new HashMap<>(2) {{
            put("isAuthorized", true);
            put("requestCode", "F45D3A4470825697-E4060300FFFBAB1F");
        }});
    }

    @GetMapping("/version")
    public Object version() {
        return result("2.0.0.1");
    }
}
