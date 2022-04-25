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

@Component
public class OnAppStarted implements ApplicationRunner {

    @Override
    public void run(ApplicationArguments args) throws IOException {
        System.out.println("Tensorflow version " + TensorFlow.version());

        var img = new Graph();
        var tf = Ops.create(img);
        var fileOp = tf.io.readFile(tf.constant("./four.jpeg"));
        var jpegOp = tf.image.decodeJpeg(fileOp.contents(), DecodeJpeg.channels(1L));
        var floatOp = tf.dtypes.cast(jpegOp, TFloat32.class);
        var normalOp = tf.math.div(floatOp, tf.constant(255.0f));
        var image = new Session(img).runner().fetch(normalOp).run().get(0);

        var graph = new Graph();
        var pbBytes = IoUtil.readBytes(Files.newInputStream(Paths.get("./model.pb")));
        graph.importGraphDef(GraphDef.parseFrom(pbBytes));
        var session = new Session(graph);
        var result = session.runner().feed("Input", image).fetch("Identity").run().get(0);

        var resultF = new float[10];
        result.asRawTensor().data().asFloats().read(resultF);

        for (int i = 0; i < 10; i++) System.out.println("number: " + i + " probability: " + resultF[i]);
    }
}
