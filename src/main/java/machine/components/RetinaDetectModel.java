package machine.components;

import com.google.protobuf.InvalidProtocolBufferException;
import machine.extend.AutoCloseMat;
import machine.helper.TensorflowHelper;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Component;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.List;

//@Component
public class RetinaDetectModel extends BaseModel {

    public Session s;
    private int IMG_SIZE = 320;
    private List<List> anchors;

    public RetinaDetectModel() throws InvalidProtocolBufferException {
        super("models/rdetect.pb");
        InitAnchor();
    }

    private void InitAnchor() {
        anchors = new ArrayList<>();
        int[] steps = {8, 16, 32};
        int[][] minsizes = {{8, 16}, {32, 64}, {128, 256}};
        int[][] feature_maps = new int[3][2];
        int total = 0;

        for (int i = 0; i < 3; i++) {
            feature_maps[i][0] = IMG_SIZE / steps[i];
            feature_maps[i][1] = IMG_SIZE / steps[i];
            total += (feature_maps[i][0] * feature_maps[i][1] * 2);
        }

        for (int i = 0; i < 3; i++) {
            for (int r = 0; r < feature_maps[i][0]; r++) {
                for (int c = 0; c < feature_maps[i][1]; c++) {
                    for (int j = 0; j < 2; j++) {
                        List tmp = new ArrayList<Float>();
                        tmp.add((float) minsizes[i][j] / IMG_SIZE);
                        tmp.add((float) minsizes[i][j] / IMG_SIZE);
                        tmp.add((float) (c + 0.5) * steps[i] / IMG_SIZE);
                        tmp.add((float) (r + 0.5) * steps[i] / IMG_SIZE);
                        anchors.add(tmp);
                    }
                }
            }
        }
    }

    public AutoCloseMat carPlateDetect(AutoCloseMat resizeImage) {
        Imgcodecs.imwrite("./resize.jpg", resizeImage);
        var IMG_SIZE = 320;
        var image = TensorflowHelper.openCVImage2Tensor(resizeImage, 1);
        List<Tensor> result = s.runner().feed("input_1", image).fetch("output_1").fetch("output_2").run();
        try (
            var confTensor = result.get(0);
            var cornerTensor = result.get(1)
        ) {
            var shape = confTensor.shape().asArray();
            var confidence_map = new float[(int) shape[1] * (int) shape[2]];
            confTensor.asRawTensor().data().asFloats().read(confidence_map);
            float max_conf = 0;
            int max_index = 0;
            for (int i = 0; i < shape[1]; i++) {
                if (confidence_map[i * 2 + 1] > max_conf) {
                    max_conf = confidence_map[i * 2 + 1];
                    max_index = i;
                }
            }
            System.out.println(max_conf);
            var plateImage = new AutoCloseMat(144, 40, CvType.CV_8UC3);
            if (max_conf > 0.7) {
                shape = cornerTensor.shape().asArray();
                var corner_map = new float[(int) shape[1] * (int) shape[2]];
                cornerTensor.asRawTensor().data().asFloats().read(corner_map);

                var points = new ArrayList<Point>();
                for (int j = 0; j < 4; j++) {
                    float x = corner_map[max_index * 8 + 2 * j] * 0.1f * (float) (anchors.get(max_index).get(0)) +
                        (float) anchors.get(max_index).get(2);
                    float y =
                        corner_map[max_index * 8 + 2 * j + 1] * 0.1f * (float) anchors.get(max_index).get(1) +
                            (float) anchors.get(max_index).get(3);
                    x = Math.min(Math.max(x, 0.0f), 1.0f) * (float) IMG_SIZE;
                    y = Math.min(Math.max(y, 0.0f), 1.0f) * (float) IMG_SIZE;
                    points.add(new Point(x, y));
                }
                var transform = Imgproc.getPerspectiveTransform(
                    new MatOfPoint2f(points.get(0), points.get(1), points.get(2), points.get(3)),
                    new MatOfPoint2f(new Point(0, 0), new Point(144, 0), new Point(0, 40), new Point(144, 40))
                );
                Imgproc.warpPerspective(resizeImage, plateImage, transform, new Size(144, 40));
//            Imgcodecs.imwrite("./tmp.jpg", plateImage);
            }
            return plateImage;
        }
    }
}
