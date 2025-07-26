import "dotenv/config";

import { DevelopmentTeamManager, DevelopmentTeamManagerInvocationTask } from "./teams/development/team-manager";

(async () => {
    const teamManager = await new DevelopmentTeamManager(true).init();
    console.log("Team manager initialized");

    console.log("Invoking add feature task");
    await teamManager.invoke({
        task: DevelopmentTeamManagerInvocationTask.AddFeature,
        data: {
            description: "Add a code reviewing agent to the development team",
        },
    }).catch((error) => {
        console.error("Error invoking add feature task:", error);
    });
    console.log("Add feature task complete");

    await teamManager.dispose();
})();