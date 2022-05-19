package machine.components;

import com.google.protobuf.InvalidProtocolBufferException;
import machine.extend.AutoCloseMat;
import machine.helper.TensorflowHelper;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Component;
import org.tensorflow.Session;

@Component
public class DetectModel {

    public Session s;

    public DetectModel() throws InvalidProtocolBufferException {
        this.s = TensorflowHelper.model("models/detect.pb");
    }

    public AutoCloseMat carPlateDetect(AutoCloseMat resizeImage) {
        var IMG_SIZE = 625;
        try (
            resizeImage;
            var image = TensorflowHelper.openCVImage2Tensor(resizeImage);
            var resultTensor = s.runner().feed("Input", image).fetch("Identity").run().get(0)
        ) {
            var coordinates = new float[8];
            resultTensor.asRawTensor().data().asFloats().read(coordinates);

            for (var i = 0; i < coordinates.length; i++) coordinates[i] *= IMG_SIZE;

            var leftTop = new Point(coordinates[0], coordinates[1]);
            var rightTop = new Point(coordinates[6], coordinates[7]);
            var leftBottom = new Point(coordinates[2], coordinates[3]);
            var rightBottom = new Point(coordinates[4], coordinates[5]);

            var transform = Imgproc.getPerspectiveTransform(
                new MatOfPoint2f(leftTop, rightTop, leftBottom, rightBottom),
                new MatOfPoint2f(new Point(0, 0), new Point(144, 0), new Point(0, 40), new Point(144, 40))
            );

            var plateImage = new AutoCloseMat(144, 40, CvType.CV_8UC3);
            Imgproc.warpPerspective(resizeImage, plateImage, transform, new Size(144, 40));
            Imgcodecs.imwrite("./tmp.jpg", plateImage);
            return plateImage;
        }
    }
}
