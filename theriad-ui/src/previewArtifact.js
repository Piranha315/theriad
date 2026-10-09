const TAILWIND_CLASS_RE = /\b(flex|grid|bg-|text-|rounded|shadow|p-\d|px-|py-|m-\d|mx-|my-|w-\d|h-\d|border-|justify-|items-|gap-|space-|min-w|max-w|overflow-)/;

// Only these tags are reasonable roots for an HTML fragment. Keeping the
// list intentional prevents a comparison such as "a <p> means..." from
// turning an otherwise normal answer into a preview.
const HTML_ROOT_TAG_RE = /<(div|section|main|button|form|article|header|footer|nav|aside|p|h[1-6]|ul|ol|table|figure|label|details|summary|blockquote|svg|canvas|video|img|input|textarea|select)(?=[\s/>])/i;
const VOID_HTML_TAGS = new Set([
  'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
  'meta', 'param', 'source', 'track', 'wbr',
]);
const CODE_FENCE_RE = /```[^\r\n]*\r?\n[\s\S]*?(?:\r?\n)?```/g;

export function hasTailwindClasses(value) {
  return TAILWIND_CLASS_RE.test(value);
}

/**
 * Return the contents of the first fenced block whose language is allowed.
 * Explicit language fences are the safe signal for rendering code that is
 * not itself an HTML fragment (for example CSS paired with a legacy preview).
 */
export function extractFencedCode(output, languages) {
  if (!output || !languages?.length) return undefined;
  const languagePattern = languages.join('|');
  const fence = new RegExp(
    '```(?:' + languagePattern + ')\\s*\\r?\\n([\\s\\S]*?)(?:\\r?\\n)?[ \\t]*```',
    'i'
  );
  return fence.exec(output)?.[1]?.trim() || undefined;
}

// Preserve indexes while masking all fenced code. This makes raw HTML
// detection ignore JSX/JavaScript/Python examples but lets us slice the exact
// source substring from the original response.
function maskCodeFences(output) {
  return output.replace(CODE_FENCE_RE, block => block.replace(/[^\r\n]/g, ' '));
}

function findTagEnd(text, start) {
  let quote = null;
  for (let index = start + 1; index < text.length; index += 1) {
    const char = text[index];
    if (quote) {
      if (char === quote) quote = null;
    } else if (char === '"' || char === "'") {
      quote = char;
    } else if (char === '>') {
      return index + 1;
    }
  }
  return -1;
}

function readHtmlToken(text, from) {
  let start = text.indexOf('<', from);
  while (start >= 0) {
    if (text.startsWith('<!--', start)) {
      const commentEnd = text.indexOf('-->', start + 4);
      if (commentEnd < 0) return null;
      return { index: start, end: commentEnd + 3, name: null };
    }

    const end = findTagEnd(text, start);
    if (end < 0) return null;
    const raw = text.slice(start, end);
    const match = raw.match(/^<\s*(\/?)\s*([A-Za-z][\w:-]*)\b/);
    if (match) {
      return {
        index: start,
        end,
        name: match[2].toLowerCase(),
        closing: Boolean(match[1]),
        selfClosing: /\/\s*>$/.test(raw),
      };
    }
    start = text.indexOf('<', end);
  }
  return null;
}

function findBalancedFragmentEnd(maskedOutput, start) {
  const stack = [];
  let cursor = start;

  while (cursor < maskedOutput.length) {
    const token = readHtmlToken(maskedOutput, cursor);
    if (!token) return null;

    if (token.name) {
      if (token.closing) {
        const matchingIndex = stack.lastIndexOf(token.name);
        if (matchingIndex >= 0) stack.splice(matchingIndex, 1);
      } else if (!token.selfClosing && !VOID_HTML_TAGS.has(token.name)) {
        stack.push(token.name);

        // Avoid interpreting </script> or </style> inside a string as real
        // markup while looking for the fragment boundary.
        if (token.name === 'script' || token.name === 'style') {
          const closeRe = new RegExp('</\\s*' + token.name + '\\b[^>]*>', 'ig');
          closeRe.lastIndex = token.end;
          const close = closeRe.exec(maskedOutput);
          if (!close) return null;
          cursor = close.index;
          continue;
        }
      }
    }

    cursor = token.end;
    if (stack.length === 0) return token.end;
  }
  return null;
}

function extractRawHtmlFragment(output, maskedOutput) {
  const firstRoot = HTML_ROOT_TAG_RE.exec(maskedOutput);
  if (!firstRoot || firstRoot.index == null) return null;

  const start = firstRoot.index;
  let end = findBalancedFragmentEnd(maskedOutput, start);
  if (end == null) return null;

  // Include adjacent root elements, but stop before ordinary prose after the
  // fragment. This supports fragments such as <header>...</header><main>...</main>.
  let cursor = end;
  while (cursor < maskedOutput.length) {
    const whitespace = maskedOutput.slice(cursor).match(/^\s*/)?.[0] ?? '';
    const nextStart = cursor + whitespace.length;
    const nextRoot = HTML_ROOT_TAG_RE.exec(maskedOutput.slice(nextStart));
    if (!nextRoot || nextRoot.index !== 0) break;
    const nextEnd = findBalancedFragmentEnd(maskedOutput, nextStart);
    if (nextEnd == null) break;
    end = nextEnd;
    cursor = nextEnd;
  }

  return {
    html: output.slice(start, end).trim(),
    start,
    end,
  };
}

function extractFullHtmlDocument(output, maskedOutput) {
  const match = maskedOutput.match(/<!doctype\s+html[\s\S]*?<\/html\s*>/i) ||
    maskedOutput.match(/<html[\s>][\s\S]*?<\/html\s*>/i);
  if (!match || match.index == null) return null;
  return output.slice(match.index, match.index + match[0].length).trim();
}

/**
 * Extract a renderable HTML/CSS artifact from model output.
 *
 * Explicit HTML fences and complete raw HTML documents are authoritative. Raw
 * fragments are only accepted when a balanced root can be isolated outside
 * every fenced code block; otherwise the caller should render the answer as
 * normal Markdown instead of presenting a misleading preview.
 */
export function extractPreviewArtifact(output) {
  if (!output || !output.trim()) return null;

  const warnings = [];
  const fencedHtml = extractFencedCode(output, ['html', 'htm', 'xhtml']);
  const css = extractFencedCode(output, ['css']);

  if (fencedHtml) {
    const isFullDoc = /<!doctype\s+html/i.test(fencedHtml) || /<html[\s>]/i.test(fencedHtml);
    return {
      kind: isFullDoc ? 'html' : hasTailwindClasses(fencedHtml) ? 'tailwind_html' : 'html_fragment',
      html: fencedHtml,
      css,
      source: 'code_fence',
      warnings,
    };
  }

  const maskedOutput = maskCodeFences(output);
  const fullDocument = extractFullHtmlDocument(output, maskedOutput);
  if (fullDocument) {
    return { kind: 'html', html: fullDocument, css, source: 'full_document', warnings };
  }

  const fragment = extractRawHtmlFragment(output, maskedOutput);
  if (!fragment) return null;

  if (output.slice(0, fragment.start).trim() || output.slice(fragment.end).trim()) {
    warnings.push('HTML fragment extracted; surrounding response text was omitted.');
  }
  return {
    kind: hasTailwindClasses(fragment.html) ? 'tailwind_html' : 'html_fragment',
    html: fragment.html,
    css,
    source: 'heuristic',
    warnings,
  };
}

export function buildPreviewDoc(artifact, tailwindEnabled) {
  const { html, css, kind } = artifact;
  const isFullDoc = kind === 'html' && (/<!doctype\s+html/i.test(html) || /<html[\s>]/i.test(html));
  // Deliberately split the tag so Vite does not interpret it in this module.
  const twScript = '<scr' + 'ipt src="https://cdn.tailwindcss.com"></scr' + 'ipt>';
  // Scripts are always allowed. They used to be gated on `tailwindEnabled`,
  // which made a checkbox labelled "Tailwind CDN" the de-facto master switch
  // for JavaScript: with it off there was no script-src at all, so
  // `default-src 'none'` blocked every script and any generated app rendered
  // its <noscript> fallback instead of running. Tailwind is a styling choice,
  // so it now only controls whether the CDN host is reachable.
  //
  // This stays safe because of what is NOT granted: the iframe carries
  // `sandbox="allow-scripts"` WITHOUT `allow-same-origin`, so the document is
  // an opaque origin that cannot touch the parent DOM, cookies, or app
  // storage; and `connect-src 'none'` still blocks network exfiltration.
  // Never add `allow-same-origin` alongside `allow-scripts` — together they
  // let the frame remove its own sandbox.
  const csp = tailwindEnabled
    ? `default-src 'none'; style-src 'unsafe-inline' https://cdn.tailwindcss.com; script-src 'unsafe-inline' https://cdn.tailwindcss.com; img-src data: blob:; connect-src 'none'`
    : `default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'`;
  const cspTag = `<meta http-equiv="Content-Security-Policy" content="${csp}" />`;

  if (isFullDoc) {
    let doc = html;
    if (css) doc = doc.replace(/<\/head>/i, `<style>\n${css}\n</style>\n</head>`);
    if (tailwindEnabled) doc = doc.replace(/<\/head>/i, `${twScript}\n</head>`);
    if (/<head[\s>]/i.test(doc)) {
      doc = doc.replace(/<head([^>]*)>/i, (match) => `${match}\n  ${cspTag}`);
    } else {
      doc = doc.replace(/<html([^>]*)>/i, (match) => `${match}<head>${cspTag}</head>`);
    }
    return doc;
  }

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  ${cspTag}
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  ${tailwindEnabled ? twScript : ''}
  ${css ? `<style>\n${css}\n</style>` : ''}
</head>
<body style="margin:0;padding:1rem;">
${html}
</body>
</html>`;
}
