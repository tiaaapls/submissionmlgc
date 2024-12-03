const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const classes = ['Cancer', 'Non-cancer'];

    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    const label = classes[classResult];

    let suggestion;

    if (confidenceScore > 50 && label === 'Cancer') {
      suggestion = "Segera periksa ke dokter!";
    }
    else if (confidenceScore <= 50 && label === 'Non-cancer') {
      suggestion = "Penyakit kanker tidak terdeteksi.";
    }
    else {
      suggestion = "Penyakit kanker tidak terdeteksi.";
    }

    return { label, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;
