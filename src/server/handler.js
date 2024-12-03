const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const storeData = require('../services/storeData');
const { Firestore } = require('@google-cloud/firestore');
const InputError = require('../exceptions/InputError');

async function postPredictHandler(request, h) {
  const { image } = request.payload;

  if (image.length > 1000000) {
    return h.response({
      status: 'fail',
      message: 'Payload content length greater than maximum allowed: 1000000'
    }).code(413);
  }

  const { model } = request.server.app;

  try {
    const { label, suggestion } = await predictClassification(model, image);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();

    const data = {
      id,
      result: label,
      suggestion,
      createdAt
    };

    await storeData(id, data);

    return h.response({
      status: 'success',
      message: 'Model is predicted successfully',
      data
    }).code(201);
  } catch (error) {
    if (error instanceof InputError) {
      return h.response({
        status: 'fail',
        message: 'Terjadi kesalahan dalam melakukan prediksi'
      }).code(400);
    }
    return h.response({
      status: 'fail',
      message: 'Gagal memproses gambar untuk prediksi'
    }).code(500);
  }
}

async function getHistoriesHandler(request, h) {
  const db = new Firestore();
  const predictCollection = db.collection('prediction');

  try {
    const snapshot = await predictCollection.get();
    const histories = snapshot.docs.map((doc) => ({
      id: doc.id,
      history: doc.data(),
    }));

    return h.response({
      status: 'success',
      data: histories,
    });
  } catch (error) {
    return h.response({
      status: 'fail',
      message: 'Gagal mengambil data riwayat prediksi.',
    }).code(500);
  }
}

module.exports = { postPredictHandler, getHistoriesHandler };
