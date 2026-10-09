// Idle "musings" shown by the thinking indicator when the model is genuinely
// working with no fresh status to report. A shuffled mix of short attributed
// quotes and Theriad-specific tips keeps the wait calm and useful.
//
// Quotes: short, attributed excerpts. Thoreau and Emerson are public domain;
// the rest are brief single-line quotations used with attribution.

export const THINKING_QUOTES = [
  {
    text: 'It is good to have an end to journey toward; but it is the journey that matters, in the end.',
    author: 'Ursula K. Le Guin',
  },
  {
    text: 'If you have built castles in the air, your work need not be lost; that is where they should be. Now put the foundations under them.',
    author: 'Henry David Thoreau',
  },
  {
    text: "You can, you should, and if you're brave enough to start, you will.",
    author: 'Stephen King',
  },
  {
    text: 'The future belongs to those who believe in the beauty of their dreams.',
    author: 'Eleanor Roosevelt',
  },
  {
    text: "You just can't beat the person who never gives up.",
    author: 'Babe Ruth',
  },
  {
    text: 'We aim above the mark to hit the mark.',
    author: 'Ralph Waldo Emerson',
  },
];

export const THINKING_TIPS = [
  'Theriad routes each task automatically using capability, uncertainty, cost, and context.',
  'Attach files to a project so Theriad can ground its answers in your own documents.',
  'Type @ in the composer to pull a skill into the conversation.',
  'Open "Show details" on an agent run to see every step, URL, and command it took.',
  'Quick chat needs no project. Pick one when you want persistent memory and files.',
  'Theriad estimates cost before running a premium model, so there are no surprises.',
  'Agent actions that touch your system pause for your approval before they run.',
  'Browser tasks hand control back to you when a CAPTCHA appears, then pick up where they left off.',
];

// One flat deck: quotes and tips interleaved by the shuffler at draw time.
export const THINKING_MUSINGS = [
  ...THINKING_QUOTES.map(q => ({ kind: 'quote', text: q.text, author: q.author })),
  ...THINKING_TIPS.map(text => ({ kind: 'tip', text })),
];

function shuffle(items) {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// Draw the next musing from a persistent, reshuffling deck. `deckRef` holds the
// remaining shuffled items so we cycle through the whole set (no immediate
// repeats) before reshuffling.
export function nextMusing(deckRef) {
  if (!deckRef.current || deckRef.current.length === 0) {
    let deck = shuffle(THINKING_MUSINGS);
    // Avoid repeating the item that was showing when the deck ran out.
    if (deckRef.last && deck.length > 1 && deck[deck.length - 1].text === deckRef.last.text) {
      [deck[0], deck[deck.length - 1]] = [deck[deck.length - 1], deck[0]];
    }
    deckRef.current = deck;
  }
  const item = deckRef.current.pop();
  deckRef.last = item;
  return item;
}

// Build a human, one-line synopsis of what an agent run is doing right now from
// the richest signal available in its trace. Returns null when there is nothing
// specific to say, letting the indicator fall back to its label or to musings.
export function agentSynopsis(run, trace = {}) {
  if (trace.progress_message) return trace.progress_message;

  const host = (url) => {
    try { return new URL(url).hostname.replace(/^www\./, ''); }
    catch { return url; }
  };

  const phase = run?.phase;
  if (phase === 'planning') return 'Mapping out an approach';
  if (phase === 'running_browser') {
    return trace.current_url ? `Reading ${host(trace.current_url)}` : 'Navigating the web';
  }
  if (phase === 'running_command') {
    const commands = trace.commands ?? [];
    const last = commands[commands.length - 1];
    const bin = last?.command ? String(last.command).trim().split(/\s+/)[0] : null;
    return bin ? `Running ${bin}` : 'Running a command';
  }

  const actions = trace.actions ?? [];
  const last = actions[actions.length - 1];
  if (last?.summary) return last.summary;
  if (last?.action) {
    return String(last.action).replace(/[_-]+/g, ' ').replace(/^\w/, c => c.toUpperCase());
  }
  return null;
}
