import { PageHeader } from "@/components/page-header";
import { PlaceholderPane } from "@/components/placeholder-pane";

export default function RunsPage() {
  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Runs"
        description="Training, calibration, and evaluation jobs — active and recent."
      />
      <PlaceholderPane resource="Runs" next="P1.D slice." />
    </div>
  );
}
