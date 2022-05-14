package machine;

import org.opencv.core.Mat;

public class AutoCloseMat extends Mat implements AutoCloseable {

    public AutoCloseMat() {
    }

    public AutoCloseMat(int rows, int cols, int type) {
        super(rows, cols, type);
    }

    @Override
    public void close() {
        release();
    }
}
