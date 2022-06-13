package machine.helper;

import cn.hutool.core.io.IoUtil;
import machine.extend.AutoCloseMat;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.web.multipart.MultipartFile;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;

import java.io.IOException;

public class TensorflowHelper {

    // 返回两个图片，0号元素是缩放图片，1号元素是原始图片
    public static FromToImageResult formToImage(MultipartFile file, int width, int height) throws IOException {
        var fileInputStream = file.getInputStream();
        var bytes = IoUtil.readBytes(fileInputStream);
        try (
            fileInputStream;
            var byteMap = new AutoCloseMat(1, bytes.length, CvType.CV_8UC1);
        ) {
            byteMap.put(0, 0, bytes);
            var mat = Imgcodecs.imdecode(byteMap, Imgcodecs.IMREAD_COLOR);

            var resized = new AutoCloseMat();
            Imgproc.resize(mat, resized, new Size(width, height));

            return new FromToImageResult(resized, mat);
        }
    }

    public static Tensor openCVImage2Tensor(Mat image, float div) {
        var channel = image.channels();

        var imageBytes = new byte[(int) (image.total() * channel)];
        image.get(0, 0, imageBytes);

        var height = image.height();
        var width = image.width();
        var imageNdArray = NdArrays.wrap(Shape.of(height, width, channel), DataBuffers.of(imageBytes, true, true));

        var tf = Ops.create();
        var floatOp = tf.dtypes.cast(tf.constant(imageNdArray), TFloat32.class);
        var normalOp = tf.math.div(floatOp, tf.constant(div));
        return tf.reshape(normalOp, tf.array(1, height, width, channel)).asTensor();
    }

    public static class FromToImageResult {

        public AutoCloseMat resized;

        public Mat raw;

        public FromToImageResult(AutoCloseMat resized, Mat raw) {
            this.resized = resized;
            this.raw = raw;
        }
    }
}
