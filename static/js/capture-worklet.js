/* AudioWorklet that forwards microphone samples to the page in 2048-sample blocks. */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(2048);
    this.n = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (input && input.length) {
      const channels = input.length;
      const len = input[0].length;
      for (let i = 0; i < len; i++) {
        let v = 0;
        for (let c = 0; c < channels; c++) v += input[c][i];
        const mono = v / channels;
        this.block[this.n++] = mono;
        if (this.n === this.block.length) {
          this.port.postMessage(this.block.slice(0));
          this.n = 0;
        }
        if (output && output.length) {
          for (let oc = 0; oc < output.length; oc++) {
            output[oc][i] = mono;
          }
        }
      }
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
