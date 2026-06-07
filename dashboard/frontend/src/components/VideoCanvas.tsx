import { useEffect, useRef } from "react";

interface Props {
  frameB64: string | null;
  width?: number;
  height?: number;
}

export function VideoCanvas({ frameB64, width = 960, height = 540 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frameB64) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        ctx.drawImage(img, 0, 0, width, height);
      });
    };
    img.src = `data:image/jpeg;base64,${frameB64}`;

    return () => cancelAnimationFrame(rafRef.current);
  }, [frameB64, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg bg-black w-full max-w-[960px]"
    />
  );
}
