package machine;

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
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

@Component
public class OnAppStarted implements ApplicationRunner {

    @Override
    public void run(ApplicationArguments args) throws IOException {
        System.out.println("Tensorflow version " + TensorFlow.version());

        numberRecognize();

        carPlateRecognize();
    }

    private void numberRecognize() throws IOException {
        var tf = Ops.create();
        var fileOp = tf.io.readFile(tf.constant("./four.jpeg"));
        var jpegOp = tf.image.decodeJpeg(fileOp.contents(), DecodeJpeg.channels(1L));
        var floatOp = tf.dtypes.cast(jpegOp, TFloat32.class);
        var image = tf.math.div(floatOp, tf.constant(255.0f)).asTensor();

        try (
            var graph = new Graph();
            var session = new Session(graph)
        ) {
            var pbBytes = IoUtil.readBytes(Files.newInputStream(Paths.get("./model.pb")));
            graph.importGraphDef(GraphDef.parseFrom(pbBytes));
            var resultTensor = session.runner().feed("Input", image).fetch("Identity").run().get(0);

            var result = new float[10];
            resultTensor.asRawTensor().data().asFloats().read(result);
            resultTensor.close();

            for (int i = 0; i < 10; i++) System.out.println("number: " + i + " probability: " + result[i]);
        }
    }

    private void carPlateRecognize() {
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

            for (Tensor t : resultTensor) {
                System.out.println(t.shape());
            }
        } catch (Exception e) {
            System.err.println(Arrays.toString(e.getStackTrace()));
        }
    }
}
