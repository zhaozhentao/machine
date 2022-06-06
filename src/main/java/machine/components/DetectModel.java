package machine.components;

import cn.hutool.core.io.resource.ResourceUtil;
import com.google.protobuf.InvalidProtocolBufferException;
import machine.entity.DetectResult;
import machine.extend.AutoCloseMat;
import machine.helper.TensorflowHelper;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.GraphDef;

import javax.annotation.Resource;
import java.util.concurrent.CountDownLatch;

@Component
public class DetectModel {

    @Resource(name = "detect_pool_thread")
    private ThreadPoolTaskExecutor executor;

    private final Session leftTop;
    private final Session leftBottom;
    private final Session rightTop;
    private final Session rightBottom;

    private Session pathToSession(String path) throws InvalidProtocolBufferException {
        var graph = new Graph();
        graph.importGraphDef(GraphDef.parseFrom(ResourceUtil.readBytes(path)));
        return new Session(graph);
    }

    public DetectModel() throws InvalidProtocolBufferException {
        leftTop = pathToSession("models/detect_left_top.pb");
        leftBottom = pathToSession("models/detect_left_bottom.pb");
        rightBottom = pathToSession("models/detect_right_bottom.pb");
        rightTop = pathToSession("models/detect_right_top.pb");
    }

    public DetectResult carPlateDetect(AutoCloseMat[] images) throws InterruptedException {
        var resizeImage = images[0];
        var rawImage = images[1];
        var image = TensorflowHelper.openCVImage2Tensor(resizeImage);
        var tensors = new Tensor[4];

        var latch = new CountDownLatch(4);
        submit(tensors, 0, leftTop, image, latch);
        submit(tensors, 1, leftBottom, image, latch);
        submit(tensors, 2, rightBottom, image, latch);
        submit(tensors, 3, rightTop, image, latch);
        latch.await();

        try (
            resizeImage;
            rawImage;
            image;
            var leftTopResult = tensors[0];
            var leftBottomResult = tensors[1];
            var rightBottomResult = tensors[2];
            var rightTopResult = tensors[3]
        ) {
            var width = rawImage.width();
            var height = rawImage.height();
            var leftTop = tensorToCoordinates(leftTopResult, width, height);
            var leftBottom = tensorToCoordinates(leftBottomResult, width, height);
            var rightBottom = tensorToCoordinates(rightBottomResult, width, height);
            var rightTop = tensorToCoordinates(rightTopResult, width, height);

            var transform = Imgproc.getPerspectiveTransform(
                new MatOfPoint2f(leftTop, rightTop, leftBottom, rightBottom),
                new MatOfPoint2f(new Point(0, 0), new Point(144, 0), new Point(0, 40), new Point(144, 40))
            );

            var plateImage = new AutoCloseMat(144, 40, CvType.CV_8UC3);
            Imgproc.warpPerspective(rawImage, plateImage, transform, new Size(144, 40));
            Imgcodecs.imwrite("./tmp.jpg", plateImage);

            return new DetectResult(leftTop, rightTop, leftBottom, rightBottom, plateImage);
        }
    }

    private void submit(Tensor[] tensors, int i, Session session, Tensor image, CountDownLatch latch) {
        executor.submit(() -> {
            tensors[i] = session.runner().feed("Input", image).fetch("Identity").run().get(0);
            latch.countDown();
        });
    }

    private Point tensorToCoordinates(Tensor leftTopResult, int width, int height) {
        var IMG_SIZE = 320;
        var coordinates = new float[2];

        leftTopResult.asRawTensor().data().asFloats().read(coordinates);

        for (var i = 0; i < coordinates.length; i++) coordinates[i] *= IMG_SIZE;

        return new Point(coordinates[0] * width / IMG_SIZE, coordinates[1] * height / IMG_SIZE);
    }
}
