import type { Item } from "../queue";
import { splitTag } from "../tags";

function summary(item: Item): string {
  switch (item.status) {
    case "queued":
      return "Waiting";
    case "uploading":
      return "Identifying";
    case "error":
      return item.error ?? "Failed";
    case "done": {
      const top = item.result?.tags.character[0];
      return top ? splitTag(top.tag).name : "No character";
    }
  }
}

interface Props {
  items: Item[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

export function QueueList({ items, selectedId, onSelect, onRemove, onClear }: Props) {
  return (
    <nav className="queue" aria-label="Images">
      <ul>
        {items.map((item) => (
          <li key={item.id} className={item.id === selectedId ? "selected" : undefined}>
            <button
              type="button"
              className="queue-item"
              aria-current={item.id === selectedId ? "true" : undefined}
              onClick={() => onSelect(item.id)}
            >
              <img src={item.url} alt="" />
              <span className="queue-text">
                <span className="queue-name">{item.name}</span>
                <span className={`queue-status ${item.status}`}>{summary(item)}</span>
              </span>
            </button>
            <button type="button" className="remove" aria-label={`Remove ${item.name}`} onClick={() => onRemove(item.id)}>
              ×
            </button>
          </li>
        ))}
      </ul>
      {items.length > 1 && (
        <button type="button" className="button secondary small clear" onClick={onClear}>
          Clear all
        </button>
      )}
    </nav>
  );
}
