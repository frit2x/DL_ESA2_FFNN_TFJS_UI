const dataset = {
  x: [],
  y: [],
  xTrain: [],
  yTrain: [],
  xTest: [],
  yTest: [],
  yTrainNoisy: [],
  yTestNoisy: []
};

const models = {};

function f(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function randomUniform(min, max) {
  return Math.random() * (max - min) + min;
}

function gaussianNoise(std) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
}

function setStatus(text) {
  const status = document.getElementById('status');
  if (status) status.textContent = text;
}

function makeDataset(N = 100, noiseVar = 0.05) {
  const xs = Array.from({ length: N }, () => randomUniform(-2, 2));
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }

  const ys = xs.map(f);
  const half = Math.floor(N / 2);

  dataset.x = xs;
  dataset.y = ys;
  dataset.xTrain = xs.slice(0, half);
  dataset.yTrain = ys.slice(0, half);
  dataset.xTest = xs.slice(half, half * 2);
  dataset.yTest = ys.slice(half, half * 2);

  const std = Math.sqrt(noiseVar);
  dataset.yTrainNoisy = dataset.yTrain.map((y) => y + gaussianNoise(std));
  dataset.yTestNoisy = dataset.yTest.map((y) => y + gaussianNoise(std));

  plotDataset();
  setStatus(`Datensatz erzeugt: N=${N}, Rauschen V=${noiseVar.toFixed(3)}`);
}

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return model;
}

function mse(yTrue, yPred) {
  let error = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const diff = yTrue[i] - yPred[i];
    error += diff * diff;
  }
  return error / yTrue.length;
}

function linspace(min, max, count) {
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

function plotDataset() {
  const traces = [
    {
      x: dataset.xTrain,
      y: dataset.yTrain,
      mode: 'markers',
      name: 'Train ohne Rauschen',
      marker: { color: '#0a57ff', size: 8 }
    },
    {
      x: dataset.xTest,
      y: dataset.yTest,
      mode: 'markers',
      name: 'Test ohne Rauschen',
      marker: { color: '#4ea5ff', size: 8 }
    },
    {
      x: dataset.xTrain,
      y: dataset.yTrainNoisy,
      mode: 'markers',
      name: 'Train mit Rauschen',
      marker: { color: '#ff4e4e', size: 8 }
    },
    {
      x: dataset.xTest,
      y: dataset.yTestNoisy,
      mode: 'markers',
      name: 'Test mit Rauschen',
      marker: { color: '#ff9a4e', size: 8 }
    }
  ];

  const layout = {
    title: 'Datensätze',
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    height: 330,
    margin: { t: 40, l: 50, r: 20, b: 40 }
  };

  Plotly.newPlot('plot-data', traces, layout, { responsive: true });
}

function plotPrediction(plotId, modelName, yTrain, yTest, title, noisy = false) {
  const xGrid = linspace(-2, 2, 250);
  const predictions = predictModel(modelName, xGrid);

  const traces = [
    {
      x: xGrid,
      y: predictions,
      mode: 'lines',
      name: 'Modellkurve',
      line: { color: '#2dbe8f', width: 3 }
    },
    {
      x: dataset.xTrain,
      y: yTrain,
      mode: 'markers',
      name: 'Train',
      marker: { color: noisy ? '#ff6f6f' : '#0a57ff', size: 8 }
    },
    {
      x: dataset.xTest,
      y: yTest,
      mode: 'markers',
      name: 'Test',
      marker: { color: noisy ? '#ffb46f' : '#4ea5ff', size: 8 }
    }
  ];

  Plotly.newPlot(plotId, traces, {
    title,
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    height: 330,
    margin: { t: 40, l: 50, r: 20, b: 40 }
  }, { responsive: true });
}

function predictModel(name, xValues) {
  const model = models[name];
  if (!model) return xValues.map(() => 0);
  const tensor = tf.tensor2d(xValues, [xValues.length, 1]);
  const preds = model.predict(tensor);
  const values = Array.from(preds.dataSync());
  tensor.dispose();
  preds.dispose();
  return values;
}

async function train(name, xData, yData, epochs = 50) {
  const model = createModel();
  models[name] = model;

  const xs = tf.tensor2d(xData, [xData.length, 1]);
  const ys = tf.tensor2d(yData, [yData.length, 1]);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize: 32,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        addLossHistory(name, logs.loss, epoch + 1);
        await tf.nextFrame();
      }
    }
  });

  xs.dispose();
  ys.dispose();
  return history;
}

const lossHistory = { clean: [], best: [], over: [] };

function addLossHistory(name, loss, epoch) {
  lossHistory[name].push({ epoch, loss });
  plotLossHistory();
}

function plotLossHistory() {
  const traces = Object.entries(lossHistory)
    .filter(([, values]) => values.length > 0)
    .map(([name, values]) => ({
      x: values.map((item) => item.epoch),
      y: values.map((item) => item.loss),
      mode: 'lines+markers',
      name,
      marker: { size: 6 }
    }));

  Plotly.newPlot('plot-losses', traces, {
    title: 'Training Loss-Verlauf (MSE)',
    xaxis: { title: 'Epoche' },
    yaxis: { title: 'MSE' },
    height: 330,
    margin: { t: 40, l: 50, r: 20, b: 40 }
  }, { responsive: true });
}

function evaluate(modelName, xData, yData) {
  const yPred = predictModel(modelName, xData);
  return mse(yData, yPred);
}

function setupEventHandlers() {
  document.getElementById('btn-generate').addEventListener('click', () => {
    const N = parseInt(document.getElementById('numPoints').value, 10);
    const V = parseFloat(document.getElementById('noiseVar').value);
    makeDataset(N, V);
  });

  document.getElementById('btn-save-data').addEventListener('click', () => {
    const data = JSON.stringify(dataset, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'dataset.json';
    link.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById('btn-load-data').addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (event) => {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const data = JSON.parse(reader.result);
        Object.assign(dataset, data);
        plotDataset();
        setStatus('Dataset geladen.');
      };
      reader.readAsText(file);
    };
    input.click();
  });

  document.getElementById('btn-save-model').addEventListener('click', async () => {
    const name = prompt('Speichere Modell unter Namen:', 'best');
    if (!name) return;
    const model = models[name];
    if (!model) {
      alert('Kein trainiertes Modell namens ' + name);
      return;
    }
    await model.save('indexeddb://' + name);
    setStatus(`Modell gespeichert: ${name}`);
  });

  document.getElementById('btn-load-model').addEventListener('click', async () => {
    const name = prompt('Lade Modell aus IndexedDB:', 'best');
    if (!name) return;
    try {
      models[name] = await tf.loadLayersModel('indexeddb://' + name);
      setStatus(`Modell geladen: ${name}`);
    } catch (error) {
      alert('Laden fehlgeschlagen: ' + error.message);
    }
  });

  document.getElementById('btn-train-clean').addEventListener('click', async () => {
    if (!dataset.xTrain.length) {
      alert('Bitte zuerst Datensatz erzeugen.');
      return;
    }
    setStatus('Trainiere Clean-Modell...');
    await train('clean', dataset.xTrain, dataset.yTrain, parseInt(document.getElementById('epochsBest').value, 10));
    const trainLoss = evaluate('clean', dataset.xTrain, dataset.yTrain);
    const testLoss = evaluate('clean', dataset.xTest, dataset.yTest);
    plotPrediction('plot-clean', 'clean', dataset.yTrain, dataset.yTest, 'Vorhersage (ohne Rauschen)');
    document.getElementById('loss-clean').textContent = `Train MSE: ${trainLoss.toExponential(3)} | Test MSE: ${testLoss.toExponential(3)}`;
    setStatus('Clean-Modell trainiert.');
  });

  document.getElementById('btn-train-best').addEventListener('click', async () => {
    if (!dataset.xTrain.length) {
      alert('Bitte zuerst Datensatz erzeugen.');
      return;
    }
    setStatus('Trainiere Best-Fit-Modell...');
    await train('best', dataset.xTrain, dataset.yTrainNoisy, parseInt(document.getElementById('epochsBest').value, 10));
    const trainLoss = evaluate('best', dataset.xTrain, dataset.yTrainNoisy);
    const testLoss = evaluate('best', dataset.xTest, dataset.yTestNoisy);
    plotPrediction('plot-best', 'best', dataset.yTrainNoisy, dataset.yTestNoisy, 'Best-Fit Vorhersage (verrauscht)', true);
    document.getElementById('loss-best').textContent = `Train MSE: ${trainLoss.toExponential(3)} | Test MSE: ${testLoss.toExponential(3)}`;
    setStatus('Best-Fit-Modell trainiert.');
  });

  document.getElementById('btn-train-over').addEventListener('click', async () => {
    if (!dataset.xTrain.length) {
      alert('Bitte zuerst Datensatz erzeugen.');
      return;
    }
    setStatus('Trainiere Overfit-Modell...');
    await train('over', dataset.xTrain, dataset.yTrainNoisy, parseInt(document.getElementById('epochsOver').value, 10));
    const trainLoss = evaluate('over', dataset.xTrain, dataset.yTrainNoisy);
    const testLoss = evaluate('over', dataset.xTest, dataset.yTestNoisy);
    plotPrediction('plot-over', 'over', dataset.yTrainNoisy, dataset.yTestNoisy, 'Overfit Vorhersage (verrauscht)', true);
    document.getElementById('loss-over').textContent = `Train MSE: ${trainLoss.toExponential(3)} | Test MSE: ${testLoss.toExponential(3)}`;
    setStatus('Overfit-Modell trainiert.');
  });
}

window.addEventListener('DOMContentLoaded', () => {
  setupEventHandlers();
  makeDataset();
});
