package machine.components;

import cn.hutool.core.io.resource.ResourceUtil;
import com.google.protobuf.InvalidProtocolBufferException;
import machine.helper.TensorflowHelper;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.proto.framework.GraphDef;

public abstract class Model {

    public Session s;

    public Model(String path) throws InvalidProtocolBufferException {
        Graph g = new Graph();
        g.importGraphDef(GraphDef.parseFrom(ResourceUtil.readBytes(path)));
        this.s = new Session(g);
    }
}
