import { useState } from "react";

import type { Tag } from "../tagger";
import type { Item } from "../queue";
import { danbooruUrl, displayTag, formatDuration, percent, ratingLabel, splitTag, tagsForCopy } from "../tags";
import { copyText } from "../clipboard";

function Match({ entry, minor }: { entry: Tag; minor?: boolean }) {
  const { name, qualifier } = splitTag(entry.tag);
  return (
    <li className={minor ? "match minor" : "match"}>
      <a className="match-name" href={danbooruUrl(entry.tag)} target="_blank" rel="noreferrer" title={entry.tag}>
        {name}
      </a>
      <span className="match-score">{percent(entry.score)}</span>
      {qualifier && <span className="match-qualifier">{qualifier}</span>}
      <span className="meter" aria-hidden="true">
        <span style={{ width: `${entry.score * 100}%` }} />
      </span>
    </li>
  );
}

function Matches({ title, tags, candidates }: { title: string; tags: Tag[]; candidates: Tag[] }) {
  return (
    <section className="block" aria-label={title}>
      <h2>{title}</h2>
      {tags.length > 0 ? (
        <ul className="matches">
          {tags.map((t) => (
            <Match key={t.tag} entry={t} />
          ))}
        </ul>
      ) : (
        <p className="muted">None found</p>
      )}
      {candidates.length > 0 && (
        <>
          <h3>Possible</h3>
          <ul className="matches">
            {candidates.map((t) => (
              <Match key={t.tag} entry={t} minor />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function Chips({ tags }: { tags: Tag[] }) {
  return (
    <ul className="chips">
      {tags.map((t) => (
        <li key={t.tag} className="chip" title={percent(t.score)}>
          {displayTag(t.tag)}
        </li>
      ))}
    </ul>
  );
}

function CopyButton({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const copy = async () => {
    setState((await copyText(text)) ? "copied" : "failed");
    setTimeout(() => setState("idle"), 1500);
  };
  return (
    <button type="button" className="button secondary small" onClick={copy}>
      {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy"}
    </button>
  );
}

const OWN_SECTION = new Set(["character", "copyright", "artist", "general"]);

export function Results({ item, onRetry }: { item: Item; onRetry: (id: string) => void }) {
  if (item.status === "error") {
    return (
      <div className="status error" role="alert">
        <span>{item.error}</span>
        {item.retryable && (
          <button type="button" className="button small" onClick={() => onRetry(item.id)}>
            Retry
          </button>
        )}
      </div>
    );
  }
  if (item.status !== "done" || !item.result) {
    return (
      <div className="status" role="status">
        <span>{item.status === "queued" ? "Waiting" : "Identifying"}</span>
        <span className="progress" aria-hidden="true" />
      </div>
    );
  }
  const r = item.result;
  const extra = Object.entries(r.tags).flatMap(([c, t]) => (OWN_SECTION.has(c) ? [] : t));
  const rating = ratingLabel(r.rating);
  return (
    <>
      <Matches title="Character" tags={r.tags.character ?? []} candidates={r.candidates.character ?? []} />
      <Matches title="Series" tags={r.tags.copyright ?? []} candidates={r.candidates.copyright ?? []} />
      {r.tags.artist && <Matches title="Artist" tags={r.tags.artist} candidates={r.candidates.artist ?? []} />}
      <section className="block" aria-label="Tags">
        <div className="block-head">
          <h2>Tags</h2>
          <CopyButton text={tagsForCopy(r)} />
        </div>
        <Chips tags={r.tags.general ?? []} />
        {extra.length > 0 && <Chips tags={extra} />}
      </section>
      <p className="facts">
        {rating && <span>Rated {rating}</span>}
        <span>
          {r.image.width} × {r.image.height}
        </span>
        <span>{formatDuration(r.timings_ms.total)}</span>
      </p>
    </>
  );
}
