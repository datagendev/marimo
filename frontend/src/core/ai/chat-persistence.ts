/* Copyright 2026 Marimo. All rights reserved. */

/**
 * Project-local persistence for the chat panel.
 *
 * Layout on disk (next to the notebook):
 *
 *     <notebook_dir>/.session/
 *       index.json           — cheap; titles + timestamps + agentSessionId
 *       <chat_id>/
 *         chat.json          — full chat body (messages, attachments, ...)
 *
 * Goals:
 *   - Popover renders instantly: only the index is fetched on mount.
 *   - Chat bodies are fetched lazily, only when the user opens one.
 *   - Writes are scoped per-chat so a typo in one chat doesn't rewrite
 *     unrelated chats and the history file is git-friendly.
 *
 * Why both layers (localStorage + disk):
 *   localStorage hydrates synchronously on first paint so the panel
 *   isn't blank during the index fetch. Disk is the source of truth
 *   that follows the notebook (across browsers, machines, server
 *   restarts) and supplies the ``agentSessionId`` the managed-agent
 *   provider needs to resume a session.
 */

import { debounce } from "lodash-es";
import { useEffect, useRef } from "react";
import { useStore } from "jotai";
import { Logger } from "@/utils/Logger";
import { getRuntimeManager } from "@/core/runtime/config";
import { type Chat, type ChatId, chatStateAtom } from "./state";

// Mounted by the DataGen fork at marimo/_server/api/endpoints/chat_history.py.
// The whole module is additive — if the backend route is absent (vanilla
// upstream marimo), every fetch below soft-fails through Logger.warn and
// the chat panel falls back to localStorage-only state.
const ROUTE = "/api/chat_history/chats";
const PER_CHAT_DEBOUNCE_MS = 500;
const ACTIVE_DEBOUNCE_MS = 200;

interface IndexEntry {
  id: ChatId;
  title: string;
  createdAt?: number;
  updatedAt?: number;
  agentSessionId?: string | null;
  messageCount?: number;
}

interface IndexResponse {
  activeChatId?: ChatId | null;
  chats?: IndexEntry[];
}

function makeUrl(path: string): string {
  return getRuntimeManager().formatHttpURL(path).toString();
}

function authHeaders(): Record<string, string> {
  return getRuntimeManager().headers() as Record<string, string>;
}

async function fetchIndex(): Promise<IndexResponse | null> {
  try {
    const r = await fetch(makeUrl(ROUTE), { headers: authHeaders() });
    if (!r.ok) {
      Logger.warn(`chat-persistence: index GET ${r.status}`);
      return null;
    }
    return (await r.json()) as IndexResponse;
  } catch (e) {
    Logger.warn("chat-persistence: index fetch failed", e);
    return null;
  }
}

async function fetchChat(chatId: ChatId): Promise<Chat | null> {
  try {
    const r = await fetch(makeUrl(`${ROUTE}/${encodeURIComponent(chatId)}`), {
      headers: authHeaders(),
    });
    if (r.status === 404) return null;
    if (!r.ok) {
      Logger.warn(`chat-persistence: chat GET ${r.status} for ${chatId}`);
      return null;
    }
    return (await r.json()) as Chat;
  } catch (e) {
    Logger.warn("chat-persistence: chat fetch failed", e);
    return null;
  }
}

async function pushChat(chat: Chat): Promise<void> {
  try {
    const r = await fetch(
      makeUrl(`${ROUTE}/${encodeURIComponent(chat.id)}`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify(chat),
      },
    );
    if (!r.ok) {
      Logger.warn(`chat-persistence: chat POST ${r.status} for ${chat.id}`);
    }
  } catch (e) {
    Logger.warn("chat-persistence: chat push failed", e);
  }
}

async function pushActive(activeChatId: ChatId | null): Promise<void> {
  try {
    await fetch(makeUrl(`${ROUTE}:active`), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ activeChatId }),
    });
  } catch (e) {
    Logger.warn("chat-persistence: active push failed", e);
  }
}

/** Build a Chat shell from an index entry (no messages yet). */
function indexEntryToStubChat(e: IndexEntry): Chat {
  return {
    id: e.id,
    title: e.title || "",
    messages: [],
    createdAt: e.createdAt ?? 0,
    updatedAt: e.updatedAt ?? 0,
  };
}

/**
 * Hook installed once at the chat-panel root.
 *
 * Effects:
 *   1. Hydrate index on mount; merge stub chats into the atom for
 *      anything the user doesn't already have locally with newer data.
 *   2. Lazy-load full chat body when ``activeChatId`` flips to a stub.
 *   3. Push per-chat writes (debounced) when a chat changes; push the
 *      active id separately so a click doesn't rewrite the chat body.
 */
export function useChatPersistence(): void {
  const store = useStore();
  // Track which chat ids we've already loaded/pushed so we don't refetch
  // or re-write the same blob on every render.
  const loadedFullBodies = useRef<Set<ChatId>>(new Set());
  const lastPushedSnapshot = useRef<Map<ChatId, Chat>>(new Map());
  const lastPushedActiveId = useRef<ChatId | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const idx = await fetchIndex();
      if (cancelled || !idx) return;
      const indexChats = idx.chats ?? [];
      if (indexChats.length === 0 && !idx.activeChatId) return;

      // Echo-suppression: pre-claim every stub we're about to insert
      // into `lastPushedSnapshot` so the writeback subscriber treats
      // the index hydration as a no-op. See the longer comment in
      // `ensureLoaded` below for the full rationale.
      //
      // The subscriber already has a `if-stub-then-skip` early-exit at
      // the bottom of this file, which currently catches index
      // hydration too (stubs have `messages: []`). We belt-and-suspend
      // here so the suppression survives a future refactor that
      // changes what a "stub" looks like.
      const cur = store.get(chatStateAtom);
      const pendingStubs = new Map<ChatId, Chat>();
      for (const entry of indexChats) {
        const localChat = cur.chats.get(entry.id);
        if (
          !localChat ||
          (entry.updatedAt ?? 0) > (localChat.updatedAt ?? 0)
        ) {
          const stub = indexEntryToStubChat(entry);
          pendingStubs.set(entry.id, stub);
          lastPushedSnapshot.current.set(entry.id, stub);
        } else {
          loadedFullBodies.current.add(entry.id);
        }
      }
      store.set(chatStateAtom, (prev) => {
        const merged = new Map(prev.chats);
        for (const [id, stub] of pendingStubs) {
          merged.set(id, stub);
        }
        return {
          ...prev,
          chats: merged,
          activeChatId: prev.activeChatId ?? idx.activeChatId ?? null,
        };
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [store]);

  // Lazy-load: when activeChatId points to a chat whose body hasn't
  // been fetched, pull just that one.
  useEffect(() => {
    const ensureLoaded = async (chatId: ChatId | null) => {
      if (!chatId) return;
      if (loadedFullBodies.current.has(chatId)) return;
      // Do we have it locally already? If messages is non-empty assume
      // it's the full thing (came from localStorage hydration).
      const cur = store.get(chatStateAtom).chats.get(chatId);
      if (cur && cur.messages.length > 0) {
        loadedFullBodies.current.add(chatId);
        return;
      }
      const full = await fetchChat(chatId);
      if (!full) {
        loadedFullBodies.current.add(chatId); // don't retry forever
        return;
      }
      // Compute the merged chat object up front so we can:
      //  (1) decide whether to mutate the atom at all, and
      //  (2) ECHO-SUPPRESS the writeback subscriber.
      //
      // Why echo-suppress:
      // The single jotai subscriber further down (`store.sub` in the
      // write-back effect) fires on every atom mutation and queues a
      // debounced POST /chats/<id> for any chat whose object reference
      // changes. That guard skips changes by comparing to
      // `lastPushedSnapshot.current` -- if a chat's ref matches the
      // last pushed snapshot, it's a no-op write and we skip.
      //
      // Without this fix:
      //   GET /chats/<id>  ->  atom.set(...new ref...)  ->  subscriber
      //   sees a "change"  ->  debounced POST /chats/<id> sends the
      //   exact bytes we just fetched BACK to the server. Pure waste:
      //   one round-trip + one S3 PUT per hydration, multiplied by the
      //   number of chats touched on AI-panel open.
      //
      // With this fix: we register the new ref into the snapshot map
      // BEFORE the atom mutation. When the subscriber runs (sync, same
      // jotai tick), `prev === chat` is true and the writeback is
      // skipped. The bytes round-trip ZERO times for the hydration
      // path. Subsequent user edits still trigger writebacks normally
      // because those edits produce a new ref that doesn't match the
      // snapshot we registered here.
      const localFull = store.get(chatStateAtom).chats.get(chatId);
      const shouldMerge =
        !localFull ||
        (full.messages?.length ?? 0) >= (localFull.messages?.length ?? 0);
      if (shouldMerge) {
        const newChat: Chat = { ...localFull, ...full };
        // Echo-suppression: pre-claim the snapshot so the writeback
        // subscriber's `prev === chat` ref-equality check skips this
        // hydration. MUST happen before `store.set` so the subscriber
        // (which runs synchronously inside set) sees the claim.
        lastPushedSnapshot.current.set(chatId, newChat);
        store.set(chatStateAtom, (prev) => {
          const merged = new Map(prev.chats);
          merged.set(chatId, newChat);
          return { ...prev, chats: merged };
        });
      }
      loadedFullBodies.current.add(chatId);
    };

    const unsubscribe = store.sub(chatStateAtom, () => {
      const v = store.get(chatStateAtom);
      void ensureLoaded(v.activeChatId);
    });
    // Run once for the initial state.
    void ensureLoaded(store.get(chatStateAtom).activeChatId);
    return () => unsubscribe();
  }, [store]);

  // Per-chat write-back. We debounce *per chat id* so editing two
  // chats in quick succession doesn't collapse to a single write.
  useEffect(() => {
    const flushers = new Map<ChatId, ReturnType<typeof debounce>>();
    const flushActive = debounce((id: ChatId | null) => {
      void pushActive(id);
    }, ACTIVE_DEBOUNCE_MS);

    function flusherFor(id: ChatId): ReturnType<typeof debounce> {
      let f = flushers.get(id);
      if (!f) {
        f = debounce((chat: Chat) => {
          void pushChat(chat);
        }, PER_CHAT_DEBOUNCE_MS);
        flushers.set(id, f);
      }
      return f;
    }

    const unsubscribe = store.sub(chatStateAtom, () => {
      const v = store.get(chatStateAtom);
      // Active id transitions
      if (v.activeChatId !== lastPushedActiveId.current) {
        lastPushedActiveId.current = v.activeChatId;
        flushActive(v.activeChatId);
      }
      // Per-chat changes
      for (const [id, chat] of v.chats) {
        const prev = lastPushedSnapshot.current.get(id);
        if (prev === chat) continue; // ref-equal: nothing to do
        // Skip pure stubs (no messages, no title, not loaded) so we
        // don't round-trip a hydration back to disk as an "edit".
        //
        // Why we ALSO check `!title` here:
        // `createNewThread` mints a Chat with `title: generateChatTitle(initialMessage)`
        // and `messages: []`, then calls `sendMessage` which kicks off
        // the AI streaming loop. The atom's `chat.messages` only gets
        // appended via `onFinish` — i.e., AFTER a successful streaming
        // turn. If the user closes the tab mid-stream or the loop errors,
        // `onFinish` never fires, the atom stays at `messages: []`,
        // and this skip used to swallow the write — so the server-side
        // `ConversationSession.title` stayed NULL even though the proxy
        // turn had already created the row from the first user prompt.
        //
        // A real "pure stub" (server hydration that we want to suppress)
        // has both empty messages AND no title for the row that has
        // never been saved before; but in practice server hydration
        // stubs are caught by the ref-equality check above because
        // `useChatPersistence`'s hydration effect pre-claims
        // `lastPushedSnapshot` with the exact stub reference BEFORE
        // mutating the atom. So this second-line defense only fires
        // for chats freshly minted by the user — and those should be
        // pushed if they carry a title, so save_chat can persist it.
        if (
          !loadedFullBodies.current.has(id) &&
          (chat.messages?.length ?? 0) === 0 &&
          (!chat.title || chat.title.length === 0)
        ) {
          continue;
        }
        lastPushedSnapshot.current.set(id, chat);
        flusherFor(id)(chat);
      }
    });
    return () => {
      unsubscribe();
      for (const f of flushers.values()) f.cancel();
      flushActive.cancel();
    };
  }, [store]);
}
