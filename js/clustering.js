/**
 * DBSCAN clustering on 2D points.
 *
 * @param {Array<{x: number, y: number}>} points
 * @param {number} eps - neighborhood radius in pixels
 * @param {number} minSamples - minimum cluster size
 * @returns {Int32Array} label per point (-1 = noise)
 */
export function dbscan(points, eps, minSamples) {
    const n = points.length;
    const labels = new Int32Array(n).fill(-1);
    const visited = new Uint8Array(n);
    const epsSq = eps * eps;
    let clusterId = 0;

    function regionQuery(idx) {
        const neighbors = [];
        const p = points[idx];
        for (let j = 0; j < n; j++) {
            const dx = p.x - points[j].x;
            const dy = p.y - points[j].y;
            if (dx * dx + dy * dy <= epsSq) neighbors.push(j);
        }
        return neighbors;
    }

    for (let i = 0; i < n; i++) {
        if (visited[i]) continue;
        visited[i] = 1;

        const neighbors = regionQuery(i);
        if (neighbors.length < minSamples) continue;

        labels[i] = clusterId;
        const queue = [...neighbors];
        const inQueue = new Uint8Array(n);
        for (const nb of neighbors) inQueue[nb] = 1;
        let qi = 0;

        while (qi < queue.length) {
            const q = queue[qi++];
            if (!visited[q]) {
                visited[q] = 1;
                const qNeighbors = regionQuery(q);
                if (qNeighbors.length >= minSamples) {
                    for (const nn of qNeighbors) {
                        if (!inQueue[nn]) {
                            queue.push(nn);
                            inQueue[nn] = 1;
                        }
                    }
                }
            }
            if (labels[q] === -1) labels[q] = clusterId;
        }
        clusterId++;
    }

    return labels;
}
