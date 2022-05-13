package machine.hooks;

import cn.hutool.core.img.ImgUtil;
import cn.hutool.core.io.IoUtil;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;

import java.awt.*;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Collectors;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;

@Component
public class OnAppStarted implements ApplicationRunner {

    private final String[] chars = new String[]{
        "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
        "Y", "Z", ""
    };

    @Override
    public void run(ApplicationArguments args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Tensorflow version " + TensorFlow.version());

        carPlateRecognize();

        carPlateDetect();
    }

    private void carPlateDetect() {
        int IMG_SIZE = 625;

        try (
            var g = new Graph();
            var s = model(g, "./models/detect.pb");
            var image = openCVImage2Tensor("./images/car.jpeg", IMG_SIZE, IMG_SIZE)
        ) {
            if (s == null) return;

            var resultTensor = s.runner().feed("Input", image).fetch("Identity").run().get(0);

            var coordinates = new float[8];
            resultTensor.asRawTensor().data().asFloats().read(coordinates);

            ArrayList<Point> points = new ArrayList<>();
            for (int i = 0; i < 4; i++)
                points.add(new Point(coordinates[2 * i] * IMG_SIZE, coordinates[2 * i + 1] * IMG_SIZE));

            // 根据x y分别找出左上 左下，右上，右下四个点
            points.sort((o1, o2) -> (int) (o1.x - o2.x));
            var lefts = points.subList(0, 2);
            lefts.sort((o1, o2) -> (int) (o1.y - o2.y));
            var leftTop = lefts.get(0);
            var leftBottom = lefts.get(1);

            var rights = points.subList(2, 4);
            rights.sort((o1, o2) -> (int) (o1.y - o2.y));
            var rightTop = rights.get(0);
            var rightBottom = rights.get(1);

            var transform = Imgproc.getPerspectiveTransform(
                new MatOfPoint2f(leftTop, rightTop, leftBottom, rightBottom),
                new MatOfPoint2f(new Point(0, 0), new Point(240, 0), new Point(0, 80), new Point(240, 80))
            );

            var src = Imgcodecs.imread("./images/car.jpeg");

            var resized = new Mat();
            Imgproc.resize(src, resized, new Size(IMG_SIZE, IMG_SIZE));

            Mat dest = new Mat(240, 80, CvType.CV_8UC3);
            Imgproc.warpPerspective(resized, dest, transform, new Size(240, 80));

            Imgcodecs.imwrite("./cut.jpg", dest);
        }
    }

    private void carPlateRecognize() {
        try (
            var g = new Graph();
            var s = model(g, "./models/plate.pb");
            var image = openCVImage2Tensor("./images/plate.jpeg", 240, 80)
        ) {
            if (s == null) return;

            var result = s.runner().
                feed("Input", image)
                .fetch("Identity")
                .fetch("Identity_1")
                .fetch("Identity_2")
                .fetch("Identity_3")
                .fetch("Identity_4")
                .fetch("Identity_5")
                .fetch("Identity_6")
                .fetch("Identity_7")
                .run();

            var plate = result.stream().map(t -> {
                var r = new float[(int) t.shape().asArray()[1]];
                t.asRawTensor().data().asFloats().read(r);
                t.close();
                return chars[findMax(r)];
            }).collect(Collectors.joining());

            System.out.println(plate);
        }
    }

    private Session model(Graph g, String path) {
        try {
            var pbBytes = IoUtil.readBytes(Files.newInputStream(Paths.get(path)));
            g.importGraphDef(GraphDef.parseFrom(pbBytes));
            return new Session(g);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    private Tensor openCVImage2Tensor(String path, int width, int height) {
        var rgb = new Mat();
        Imgproc.cvtColor(Imgcodecs.imread(path), rgb, COLOR_BGR2RGB);

        var resized = new Mat();
        Imgproc.resize(rgb, resized, new Size(width, height));
        var channel = resized.channels();

        var imageBytes = new byte[(int) (resized.total() * channel)];
        resized.get(0, 0, imageBytes);

        var imageNdArray = NdArrays.wrap(Shape.of(height, width, channel), DataBuffers.of(imageBytes, true, false));

        var tf = Ops.create();
        var floatOp = tf.dtypes.cast(tf.constant(imageNdArray), TFloat32.class);
        var normal = tf.math.div(floatOp, tf.constant(255.0f));
        var reshape = tf.reshape(normal, tf.array(1, height, width, channel));
        return reshape.asTensor();
    }

    private int findMax(float[] array) {
        if (array == null || array.length == 0) return -1;

        var largest = 0;
        for (var i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest;
    }
}
