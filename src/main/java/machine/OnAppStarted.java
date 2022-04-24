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
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.op.math.Div;
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

        byte[] pbBytes = IoUtil.readBytes(Files.newInputStream(Paths.get("./model.pb")));

        Graph img = new Graph();
        Ops tf = Ops.create(img);
        ReadFile fileOp = tf.io.readFile(tf.constant("./four.jpeg"));
        DecodeJpeg jpegOp = tf.image.decodeJpeg(fileOp.contents(), DecodeJpeg.channels(1L));
        Cast<TFloat32> floatOp = tf.dtypes.cast(jpegOp, TFloat32.class);
        Div<TFloat32> normalOp = tf.math.div(floatOp, tf.constant(255.0f));
        Tensor image = new Session(img).runner().fetch(normalOp).run().get(0);

        Graph graph = new Graph();
        graph.importGraphDef(GraphDef.parseFrom(pbBytes));
        Session session = new Session(graph);
        Tensor result = session.runner()
            .feed("Input", image)
            .fetch("Identity")
            .run()
            .get(0);

        float[] resultF = new float[10];
        result.asRawTensor().data().asFloats().read(resultF);

        for (int i = 0; i < 10; i++) {
            System.out.println("number: " + i + " probability: " + resultF[i]);
        }
    }
}
