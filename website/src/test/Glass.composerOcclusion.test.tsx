/**
 * The composer dock floats over the transcript scroller and the conversation
 * scrolls UNDER it (iOS toolbar layout); there is deliberately no opaque fade
 * band, so the dock's own `--glass-tint` (~40-45% opaque) plus its backdrop
 * blur are the ONLY thing that hides the covered strip -- pinned in
 * ChatPage.dockClearance.test.tsx.
 *
 * At frost 4 that occluder was too weak: a 4px blur leaves ~13px body glyphs
 * legible and the tint passes 55-60% of the backdrop, so a tall message
 * (Autopilot goal/stage injections are the reliable trigger) read straight
 * THROUGH the composer mid-scroll (#15225). The panel therefore runs a heavier
 * blur than the small chips, whose pills never cover a scrolling message and
 * read foggy at 30px tall under a heavy blur.
 *
 * Pinned against source text, like the sibling glass suites: RECIPE is
 * module-private and happy-dom has no layout for the primitive's ResizeObserver
 * to fire against, so the recipe the browser paints is read from the file.
 */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const GLASS_SRC = readFileSync(resolve(process.cwd(), 'src/components/Glass.tsx'), 'utf-8')

/** The frost (backdrop-blur px) each variant sets in the RECIPE map. */
function frostOf(variant: 'panel' | 'chip'): number {
  const m = new RegExp(`${variant}:\\s*\\{\\s*frost:\\s*(\\d+)`).exec(GLASS_SRC)
  expect(m, `RECIPE.${variant} frost not found`).not.toBeNull()
  return Number(m![1])
}

describe('composer glass occlusion (#15225)', () => {
  it('blurs the panel hard enough to smear body text past reading', () => {
    // The composer band is body-typography (~13px). A blur radius at least the
    // glyph height destroys the letterforms; 4px only softened them, which is
    // exactly what let a tall message read through the dock.
    expect(frostOf('panel')).toBeGreaterThanOrEqual(12)
  })

  it('keeps the small chips lighter than the panel', () => {
    // A chip pill does not cover a scrolling message, so it keeps the light
    // blur the maintainer tuned for a 30px-tall pill; only the panel needs the
    // heavier occluding blur.
    expect(frostOf('chip')).toBeLessThan(frostOf('panel'))
    expect(frostOf('chip')).toBe(4)
  })

  it('shares the light band across both variants', () => {
    // The blur split is the only difference; the band stays unified at 25.
    for (const v of ['panel', 'chip'] as const) {
      const m = new RegExp(`${v}:\\s*\\{[^}]*lightIntensity:\\s*(\\d+)`).exec(GLASS_SRC)
      expect(m, `RECIPE.${v} lightIntensity not found`).not.toBeNull()
      expect(Number(m![1])).toBe(25)
    }
  })
})
