export function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: {
  title?: string;
  message?: string;
  onRetry?: () => void;
}) {
  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50 p-6 text-sm dark:border-rose-900/50 dark:bg-rose-950/30">
      <p className="font-medium text-rose-900 dark:text-rose-200">{title}</p>
      {message ? (
        <p className="mt-1 text-rose-800 dark:text-rose-300">{message}</p>
      ) : null}
      {onRetry ? (
        <button
          onClick={onRetry}
          className="mt-3 rounded-md border border-rose-300 bg-white px-3 py-1 text-xs font-medium text-rose-900 hover:bg-rose-100 dark:border-rose-800 dark:bg-rose-950 dark:text-rose-200 dark:hover:bg-rose-900/50"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
