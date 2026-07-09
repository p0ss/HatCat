import type { HTMLAttributes } from "react";

type DivProps = HTMLAttributes<HTMLDivElement>;

export function Card({ children, className = "", ...rest }: DivProps) {
  return (
    <div
      {...rest}
      className={`rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = "", ...rest }: DivProps) {
  return (
    <div
      {...rest}
      className={`px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 ${className}`}
    >
      {children}
    </div>
  );
}

export function CardBody({ children, className = "", ...rest }: DivProps) {
  return (
    <div {...rest} className={`p-4 ${className}`}>
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  className = "",
  ...rest
}: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      {...rest}
      className={`text-sm font-medium text-zinc-900 dark:text-zinc-100 ${className}`}
    >
      {children}
    </h3>
  );
}
