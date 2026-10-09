import type { RuntimeProfile } from './types';

const SENSITIVE_TERMS = [
  'checkout',
  'purchase',
  'payment',
  'delete',
  'send email',
  'send slack',
  'send message',
  'submit form',
  'log in',
  'login',
  'oauth',
  'account connection',
  'financial transaction',
  'transfer money',
  'irreversible',
];

const ALWAYS_CONFIRM_TERMS = [
  'payment',
  'purchase',
  'checkout',
  'delete',
  'password',
  'credential',
  'send email',
  'send slack',
  'send message',
  'financial transaction',
  'transfer money',
];

export function isSensitiveInstruction(text: string, runtimeProfile: RuntimeProfile): boolean {
  const normalized = text.toLowerCase();
  const terms = runtimeProfile === 'personal_trusted' ? ALWAYS_CONFIRM_TERMS : SENSITIVE_TERMS;
  return terms.some((term) => normalized.includes(term));
}

export function confirmSensitiveInstruction(text: string, runtimeProfile: RuntimeProfile): boolean {
  if (!isSensitiveInstruction(text, runtimeProfile)) {
    return true;
  }
  return window.confirm(
    `Sensitive action check (${runtimeProfile})\n\n` +
      'This request appears to involve a sensitive or irreversible action. ' +
      'Normal Theriad chat will not execute browser actions directly.\n\n' +
      `Request:\n${text}\n\nContinue as analysis/planning only?`
  );
}

