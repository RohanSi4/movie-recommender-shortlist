"use client";

import Image from "next/image";
import { useState } from "react";

type PosterImageProps = {
  src?: string;
  alt: string;
  sizes: string;
  priority?: boolean;
};

export function PosterImage({ src, alt, sizes, priority = false }: PosterImageProps) {
  const [failed, setFailed] = useState(false);

  return (
    <Image
      src={!src || failed ? "/poster-fallback.svg" : src}
      alt={alt}
      fill
      sizes={sizes}
      priority={priority}
      onError={() => setFailed(true)}
    />
  );
}
