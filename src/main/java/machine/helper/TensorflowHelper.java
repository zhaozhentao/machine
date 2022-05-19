package machine.helper;

import cn.hutool.core.io.IoUtil;
import cn.hutool.core.io.resource.ResourceUtil;
import com.google.protobuf.InvalidProtocolBufferException;
import machine.extend.AutoCloseMat;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.web.multipart.MultipartFile;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;

import java.io.IOException;

public class TensorflowHelper {

    public static AutoCloseMat formToImage(MultipartFile file, int width, int height) throws IOException {
        var fileInputStream = file.getInputStream();
        var bytes = IoUtil.readBytes(fileInputStream);
        try (
            fileInputStream;
            var byteMap = new AutoCloseMat(1, bytes.length, CvType.CV_8UC1);
            var rgb = new AutoCloseMat()
        ) {
            byteMap.put(0, 0, bytes);
            var mat = Imgcodecs.imdecode(byteMap, Imgcodecs.IMREAD_COLOR);

            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB);
            mat.release();

            var resized = new AutoCloseMat();
            Imgproc.resize(rgb, resized, new Size(width, height));

            return resized;
        }
    }

    public static Tensor openCVImage2Tensor(Mat image) {
        var channel = image.channels();

        var imageBytes = new byte[(int) (image.total() * channel)];
        image.get(0, 0, imageBytes);

        var height = image.height();
        var width = image.width();
        var imageNdArray = NdArrays.wrap(Shape.of(height, width, channel), DataBuffers.of(imageBytes, true, true));

        var tf = Ops.create();
        var floatOp = tf.dtypes.cast(tf.constant(imageNdArray), TFloat32.class);
        var normalOp = tf.math.div(floatOp, tf.constant(255.0f));
        return tf.reshape(normalOp, tf.array(1, height, width, channel)).asTensor();
    }
}
