package machine.models;

import cn.hutool.core.io.resource.ResourceUtil;
import com.google.protobuf.InvalidProtocolBufferException;
import org.opencv.core.Core;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.proto.framework.GraphDef;

@Component
public class ModelProvider {

    public Session carPlateDetectSession;

    public Session carPlateRecognizeSession;

    public ModelProvider() throws InvalidProtocolBufferException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        this.carPlateDetectSession = model("models/detect.pb");
        this.carPlateRecognizeSession = model("models/ocr_model.pb");
    }

    private Session model(String path) throws InvalidProtocolBufferException {
        Graph g = new Graph();
        g.importGraphDef(GraphDef.parseFrom(ResourceUtil.readBytes(path)));
        return new Session(g);
    }
}
