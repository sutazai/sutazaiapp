<DiamondWindow>
    <HolographicToolbar>
        <FileDropzone on:files={handleDrop}>
            <Tabs>
                <Tab title="Documents">
                    <DocumentPreview3D {files}/>
                </Tab>
                <Tab title="Analysis">
                    <InteractiveChart data={analysisData}/>
                </Tab>
            </Tabs>
        </FileDropzone>
    </HolographicToolbar>
    
    <GlassPanel side="right">
        <AIChatInterface voiceEnabled={true} hologramMode="projection"/>
    </GlassPanel>
</DiamondWindow> 