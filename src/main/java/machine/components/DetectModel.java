package machine.components;

import com.google.protobuf.InvalidProtocolBufferException;
import machine.entity.DetectResult;
import machine.extend.AutoCloseMat;
import machine.helper.TensorflowHelper;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Component;

@Component
public class DetectModel extends BaseModel {

    public DetectModel() throws InvalidProtocolBufferException {
        super("models/detect.pb");
    }

    public DetectResult carPlateDetect(AutoCloseMat resized, Mat raw) {
        var IMG_SIZE = 320;
        var PLATE_HEIGHT = 50;
        try (
            var resizeImage = resized;
            var image = TensorflowHelper.openCVImage2Tensor(resizeImage, 255);
            var resultTensor = s.runner().feed("Input", image).fetch("Identity").run().get(0)
        ) {
            var coordinates = new float[8];
            resultTensor.asRawTensor().data().asFloats().read(coordinates);

            for (var i = 0; i < coordinates.length; i++) coordinates[i] *= IMG_SIZE;

            var width = raw.width();
            var height = raw.height();
            var leftTop = new Point(coordinates[0] * width / IMG_SIZE, coordinates[1] * height / IMG_SIZE);
            var rightTop = new Point(coordinates[6] * width / IMG_SIZE, coordinates[7] * height / IMG_SIZE);
            var leftBottom = new Point(coordinates[2] * width / IMG_SIZE, coordinates[3] * height / IMG_SIZE);
            var rightBottom = new Point(coordinates[4] * width / IMG_SIZE, coordinates[5] * height / IMG_SIZE);

            var transform = Imgproc.getPerspectiveTransform(
                new MatOfPoint2f(leftTop, rightTop, leftBottom, rightBottom),
                new MatOfPoint2f(new Point(0, 0), new Point(144, 0), new Point(0, PLATE_HEIGHT), new Point(144, PLATE_HEIGHT))
            );

            var plateImage = new AutoCloseMat(144, PLATE_HEIGHT, CvType.CV_8UC3);
            Imgproc.warpPerspective(raw, plateImage, transform, new Size(144, PLATE_HEIGHT));
            raw.release();
            Imgcodecs.imwrite("./tmp.jpg", plateImage);

            return new DetectResult(leftTop, rightTop, leftBottom, rightBottom, plateImage);
        }
    }
}
