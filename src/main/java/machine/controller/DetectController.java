package machine.controller;

import cn.hutool.core.io.IoUtil;
import machine.extend.AutoCloseMat;
import machine.models.ModelProvider;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;

import javax.annotation.Resource;
import java.io.IOException;

@RestController
public class DetectController {

    private final String[] chars = "京,沪,津,渝,冀,晋,蒙,辽,吉,黑,苏,浙,皖,闽,赣,鲁,豫,鄂,湘,粤,桂,琼,川,贵,云,藏,陕,甘,青,宁,新,0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,W,X,Y,Z,港,学,使,警,澳,挂,军,北,南,广,沈,兰,成,济,海,民,航,空".split(",");

    @Resource
    private ModelProvider p;

    @PostMapping("/car_plate")
    public Object detect(@RequestParam("file") MultipartFile file) throws IOException {
        var image = formToImage(file);

        var plateImage = carPlateDetect(image);

        return carPlateRecognize(plateImage);
    }

    private AutoCloseMat carPlateDetect(AutoCloseMat resizeImage) {
        var IMG_SIZE = 625;
        try (resizeImage) {
            long a = System.currentTimeMillis();
            var image = openCVImage2Tensor(resizeImage);
            System.out.println("openCVImage2Tensor " + (System.currentTimeMillis() - a));

            a = System.currentTimeMillis();
            var resultTensor = p.carPlateDetectSession.runner().feed("Input", image).fetch("Identity").run().get(0);
            System.out.println("detect " + (System.currentTimeMillis() - a));

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

    private Object carPlateRecognize(AutoCloseMat plateImage) {
        try (
            plateImage;
            var image = openCVImage2Tensor(plateImage);
            var result = p.carPlateRecognizeSession.runner().
                feed("input_1:0", image)
                .fetch("output_1")
                .run().get(0);
        ) {
            var shape = result.shape().asArray();

            var ndArray = NdArrays.wrap(
                Shape.of(shape[0], shape[1], shape[2]),
                result.asRawTensor().data().asFloats()
            );

            StringBuilder sb = new StringBuilder();
            for (var i = 0; i < shape[0]; i++) {
                var floatDataBuffer = DataBuffers.ofFloats(shape[2]);
                ndArray.get(i).get(0).read(floatDataBuffer);
                float[] floats = new float[(int) shape[2]];
                floatDataBuffer.read(floats);
                var index = findMax(floats);
                if (index >= chars.length) continue;
                sb.append(chars[index]);
            }

            return sb.toString();
        }
    }

    private AutoCloseMat formToImage(MultipartFile file) throws IOException {
        var fileInputStream = file.getInputStream();
        byte[] bytes = IoUtil.readBytes(fileInputStream);
        try (
            fileInputStream;
            var byteMap = new AutoCloseMat(1, bytes.length, CvType.CV_8UC1);
            var rgb = new AutoCloseMat()
        ) {
            byteMap.put(0, 0, bytes);
            Mat mat = Imgcodecs.imdecode(byteMap, Imgcodecs.IMREAD_COLOR);

            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB);
            mat.release();

            var resized = new AutoCloseMat();
            Imgproc.resize(rgb, resized, new Size(625, 625));

            return resized;
        }
    }

    private Tensor openCVImage2Tensor(Mat image) {
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

    private int findMax(float[] array) {
        if (array == null || array.length == 0) return -1;

        var largest = 0;
        for (var i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest;
    }
}
