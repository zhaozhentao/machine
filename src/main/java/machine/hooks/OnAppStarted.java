package machine.hooks;

import cn.hutool.core.io.IoUtil;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.EncodeJpeg;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

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
        System.out.println("Tensorflow version " + TensorFlow.version());

        carPlateRecognize();

        carPlateDetect();

        System.out.println("finish");
    }

    private void carPlateDetect() {
        try (
            var g = new Graph();
            var s = model(g, "./models/zc.pb");
            var image = imageTensor("./images/car.jpeg")
        ) {
            if (s == null) return;

            var result = (TFloat32) s.runner().feed("Input", image).fetch("Identity").run().get(0);
            var shape = result.shape().asArray();

            var imageGraph = new Graph();
            var imageSession = new Session(imageGraph);
            var tf = Ops.create(imageGraph);

            var unNormal = tf.math.mul(tf.constant(result), tf.constant(255.0f));
            var maskFloat = tf.reshape(unNormal, tf.array(shape[1], shape[2], shape[3]));
            var maskInt = tf.dtypes.cast(maskFloat, TUint8.class);
            var jpeg = tf.image.encodeJpeg(maskInt, EncodeJpeg.quality(100L));
            var writeFile = tf.io.writeFile(tf.constant("./mask.jpeg"), jpeg);

            imageSession.runner().addTarget(writeFile).run();
        }
    }

    private void carPlateRecognize() {
        try (
            var g = new Graph();
            var s = model(g, "./models/plate.pb");
            var image = imageTensor("./images/plate.jpeg")
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

    private Tensor imageTensor(String path) {
        try (
            var g = new Graph();
            var s = new Session(g)
        ) {
            var tf = Ops.create(g);
            var file = tf.io.readFile(tf.constant(path));
            var jpeg = tf.image.decodeJpeg(file.contents(), DecodeJpeg.channels(3L));
            var floatOp = tf.dtypes.cast(jpeg, TFloat32.class);
            var normal = tf.math.div(floatOp, tf.constant(255.0f));
            var shape = s.runner().fetch(normal).run().get(0).shape();
            var shapeArray = shape.asArray();
            var reshape = tf.reshape(normal, tf.array(1, shapeArray[0], shapeArray[1], shapeArray[2]));
            return s.runner().fetch(reshape).run().get(0);
        }
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
