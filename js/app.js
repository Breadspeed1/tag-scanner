async function waitForOpenCV() {
    while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
    }
}

let refFeatures = null;
let ocrWorker = null;

async function start() {
    await waitForOpenCV();
    console.log('opencv ready');


}


start()