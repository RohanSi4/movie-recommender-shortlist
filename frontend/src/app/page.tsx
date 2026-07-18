import { RecommenderApp } from "@/components/recommender-app";

export default function Home() {
  return (
    <div className="site-shell">
      <a className="skip-link" href="#main-content">Skip to the movies</a>
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />
      <RecommenderApp />
    </div>
  );
}
