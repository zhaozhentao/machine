package machine;

import cn.hutool.core.io.IoUtil;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

@Component
public class OnAppStarted implements ApplicationRunner {

    private final String[] chars = new String[]{
        "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
        "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
        "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", ""
    };

    @Override
    public void run(ApplicationArguments args) throws IOException {
        System.out.println("Tensorflow version " + TensorFlow.version());

        carPlateRecognize();
    }

    private void carPlateRecognize() throws IOException {
        try (
            var g = new Graph();
            var s = new Session(g);
            var graph = new Graph();
            var session = new Session(graph)
        ) {
            var tf = Ops.create(g);
            var file = tf.io.readFile(tf.constant("./plate.jpeg"));
            var jpeg = tf.image.decodeJpeg(file.contents(), DecodeJpeg.channels(3L));
            var floatOp = tf.dtypes.cast(jpeg, TFloat32.class);
            var normal = tf.math.div(floatOp, tf.constant(255.0f));
            var shape = s.runner().fetch(normal).run().get(0).shape();
            var shapeArray = shape.asArray();
            var reshape = tf.reshape(normal, tf.array(1, shapeArray[0], shapeArray[1], shapeArray[2]));
            var image = s.runner().fetch(reshape).run().get(0);
            var pbBytes = IoUtil.readBytes(Files.newInputStream(Paths.get("./plate.pb")));

            graph.importGraphDef(GraphDef.parseFrom(pbBytes));

            var resultTensor = session.runner()
                .feed("Input", image)
                .fetch("Identity")
                .fetch("Identity_1")
                .fetch("Identity_2")
                .fetch("Identity_3")
                .fetch("Identity_4")
                .fetch("Identity_5")
                .fetch("Identity_6")
                .fetch("Identity_7")
                .run();

            String plate = resultTensor.stream().map(t -> {
                float[] r = new float[(int) t.shape().asArray()[1]];
                t.asRawTensor().data().asFloats().read(r);
                t.close();
                return chars[findMax(r)];
            }).collect(Collectors.joining());

            System.out.println(plate);
        }
    }

    private int findMax(float[] array) {
        if (array == null || array.length == 0) return -1;

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest;
    }
}
