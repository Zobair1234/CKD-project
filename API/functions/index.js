const functions = require("firebase-functions");
const tf = require("@tensorflow/tfjs");

exports.api = functions.https.onRequest((req, res) => {
	res.set("Access-Control-Allow-Origin", "*");
	res.set("Access-Control-Allow-Methods", "GET, POST");

	// let data = req.query.data;
	// data = data.split(",");
	
	// let df = [];
	// data.forEach((e) => {
	// 	df.push(parseFloat(e));
	// });
	
	predict().then((pred) => {
		res.send(pred);
	});
});

async function predict() {
	const model = await tf.loadLayersModel('model.json');
	
	let data = [53.0,60.0,1.025,0.0,0.0,1.0,1.0,0.0,0.0,116.0,26.0,1.0,146.0,4.9,15.8,45.0,7700.0,5.2,0.37,0,0.09,1.0,0.0,0.0];

	let min = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0];
	let max = [120, 360, 1.1, 5.5, 5.5, 1, 1, 1, 1, 700, 500, 100, 200, 100, 50, 100, 30000, 20, 1, 1, 1, 1, 1, 1];

	for (let i = 0; i < 24; i++) {
		data[i] = (data[i] - min[i]) / (max[i] - min[i]);
	}

	return await model.predict(tf.tensor2d([data])).arraySync();
}