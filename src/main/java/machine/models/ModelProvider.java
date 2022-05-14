package machine.models;

import cn.hutool.core.io.IoUtil;
import org.opencv.core.Core;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.proto.framework.GraphDef;

import java.nio.file.Files;
import java.nio.file.Paths;

@Component
public class ModelProvider {

    public Session carPlateDetectSession;

    public Session carPlateRecognizeSession;

    public ModelProvider() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        this.carPlateDetectSession = model("./models/detect.pb");
        this.carPlateRecognizeSession = model("./models/ocr_model.pb");
    }

    private Session model(String path) {
        try {
            Graph g = new Graph();
            g.importGraphDef(GraphDef.parseFrom(IoUtil.readBytes(Files.newInputStream(Paths.get(path)))));
            return new Session(g);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }
}
