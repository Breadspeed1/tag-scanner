async function waitForOpenCV() {
    while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
    }
}

await waitForOpenCV();
console.log('OpenCV ready:', typeof cv.Mat);