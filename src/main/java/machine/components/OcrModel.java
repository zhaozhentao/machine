package machine.components;

import com.google.protobuf.InvalidProtocolBufferException;
import machine.extend.AutoCloseMat;
import machine.helper.TensorflowHelper;
import org.springframework.stereotype.Component;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;

@Component
public class OcrModel extends BaseModel {

    final String[] chars = "京,沪,津,渝,冀,晋,蒙,辽,吉,黑,苏,浙,皖,闽,赣,鲁,豫,鄂,湘,粤,桂,琼,川,贵,云,藏,陕,甘,青,宁,新,0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,W,X,Y,Z,港,学,使,警,澳,挂,军,北,南,广,沈,兰,成,济,海,民,航,空".split(",");

    public OcrModel() throws InvalidProtocolBufferException {
        super("models/ocr_model.pb");
    }

    public String carPlateRecognize(AutoCloseMat plateImage) {
        try (
            plateImage;
            var image = TensorflowHelper.openCVImage2Tensor(plateImage);
            var result = s.runner().feed("input_1:0", image).fetch("output_1").run().get(0)
        ) {
            var shape = result.shape().asArray();

            // shape is (14, 1, 87)，each plate have 14 character at most
            var ndArray = NdArrays.wrap(
                Shape.of(shape[0], shape[1], shape[2]),
                result.asRawTensor().data().asFloats()
            );

            var sb = new StringBuilder();
            for (var i = 0; i < shape[0]; i++) {
                var floatDataBuffer = DataBuffers.ofFloats(shape[2]);
                ndArray.get(i).get(0).read(floatDataBuffer);
                var floats = new float[(int) shape[2]];
                floatDataBuffer.read(floats);
                var index = findMax(floats);
                if (index >= chars.length) continue;
                sb.append(chars[index]);
            }

            return sb.toString();
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
