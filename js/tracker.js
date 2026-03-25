import { TRACKER } from './config.js';

export class CardTracker {
    constructor(config = TRACKER) {
        this.mergeRadiusSq = config.merge_radius ** 2;
        this.alpha = config.ema_alpha;
        this.cards = new Map();
        this.nextId = 0;
    }

    /**
     * Update tracker with a newly detected card.
     * Returns { id, isNew } where isNew means this card hasn't been seen before.
     */
    update(detectedCard) {
        const { center } = detectedCard;
        let bestId = null;
        let bestDistSq = Infinity;

        for (const [id, tracked] of this.cards) {
            const dx = center.x - tracked.center.x;
            const dy = center.y - tracked.center.y;
            const distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestId = id;
            }
        }

        if (bestId !== null && bestDistSq < this.mergeRadiusSq) {
            const tracked = this.cards.get(bestId);
            tracked.center.x = tracked.center.x * (1 - this.alpha) + center.x * this.alpha;
            tracked.center.y = tracked.center.y * (1 - this.alpha) + center.y * this.alpha;
            tracked.framesSeen++;
            tracked.lastSeen = performance.now();
            return { id: bestId, isNew: false };
        }

        const id = this.nextId++;
        this.cards.set(id, {
            center: { ...center },
            skuText: null,
            ocrPending: false,
            framesSeen: 1,
            lastSeen: performance.now(),
        });
        return { id, isNew: true };
    }

    setText(id, text) {
        const card = this.cards.get(id);
        if (card) {
            card.skuText = text;
            card.ocrPending = false;
        }
    }

    getText(id) {
        return this.cards.get(id)?.skuText ?? null;
    }

    setOcrPending(id) {
        const card = this.cards.get(id);
        if (card) card.ocrPending = true;
    }

    isOcrNeeded(id) {
        const card = this.cards.get(id);
        return card && !card.skuText && !card.ocrPending;
    }

    getAll() {
        return [...this.cards.entries()].map(([id, card]) => ({
            id,
            center: card.center,
            skuText: card.skuText,
            ocrPending: card.ocrPending,
            framesSeen: card.framesSeen,
        }));
    }
}
